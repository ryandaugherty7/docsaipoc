#!/usr/bin/env python3

import os
import sys
import requests
from langchain_aws import ChatBedrock
from langchain_community.embeddings import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import List, Dict
import json
from elasticsearch import Elasticsearch
import re
import boto3
import gradio as gr

# --- 1. Initialize Bedrock and Elasticsearch Clients with Authentication ---
# Configure AWS credentials for Bedrock
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print("Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables must be set")
    exit()

# Initialize Claude via Bedrock
bedrock_llm = ChatBedrock(
    model_id="arn:aws:bedrock:us-east-1:273354632795:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name=AWS_REGION,
    provider="anthropic"
)

# Keep Ollama for sentence embeddings
ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Elasticsearch client setup - using same remote ES as source
ELASTICSEARCH_URL = "https://29944f54da01413ab55da3b9f2fa68ad.us-east-1.aws.found.io:443"
INDEX_NAME = "appian_docs_rag"
ES_API_KEY = os.getenv("SOURCE_ES_API_KEY")

if not ES_API_KEY:
    print("Error: SOURCE_ES_API_KEY environment variable not set")
    exit()

# Verify connection with API key authentication
try:
    headers = {"Authorization": f"ApiKey {ES_API_KEY}"}
    requests.get(f"{ELASTICSEARCH_URL}/_cluster/health", headers=headers)
    print("Successfully connected to Elasticsearch with API key.")
except requests.exceptions.ConnectionError:
    print(f"Failed to connect to Elasticsearch at {ELASTICSEARCH_URL}. Please ensure it is running.")
    exit()

# Create Elasticsearch client for existing data
es_client = Elasticsearch(
    [ELASTICSEARCH_URL],
    api_key=ES_API_KEY,
    verify_certs=True
)

# --- 2. Custom Hybrid Search Implementation ---

def extract_keywords(query: str) -> str:
    """Extract keywords from natural language query for lexical search"""
    # Remove punctuation
    query = re.sub(r'[?!:;.]', '', query)
    # Remove question words and common phrases
    query = re.sub(r'\b(what|are|the|different|how|why|when|where|make|up|that|of|in|to|for|with|by|can|do|does|is|will|would|should|could)\b', '', query, flags=re.IGNORECASE)
    # Keep only meaningful words (3+ chars)
    words = [word.strip() for word in query.split() if len(word.strip()) > 2]
    return ' '.join(words)

def hybrid_search(query: str, k: int = 6) -> List[Dict]:
    """Perform hybrid search combining vector similarity and lexical search"""
    
    # Generate query embedding
    query_embedding = ollama_embeddings.embed_query(query)
    
    # Vector similarity query with filters
    vector_query = {
        "knn": {
            "field": "vector",
            "query_vector": query_embedding,
            "k": k * 3,
            "num_candidates": k * 10,
            "filter": {
                "bool": {
                    "must_not": [
                        {"match_phrase": {"title": {"query": "Deprecated"}}},
                        {"match": {"hideinsearch": True}},
                        {"match": {"islatestsubprodver": False}}
                    ]
                }
            }
        }
    }
    
    # Extract keywords for lexical search
    lexical_keywords = extract_keywords(query)
    
    # Lexical query with synonyms and filters
    lexical_query = {
        "query": {
            "bool": {
                "must_not": [
                    {"match_phrase": {"title": {"query": "Deprecated"}}},
                    {"match": {"hideinsearch": True}},
                    {"match": {"islatestsubprodver": False}}
                ],
                "should": [
                    {"match": {"title": {"query": lexical_keywords, "boost": 6}}},
                    {"match_phrase_prefix": {"title": {"query": lexical_keywords, "analyzer": "with_stemmer", "boost": 5}}},
                    {"match": {"headers": {"query": lexical_keywords, "boost": 4}}},
                    {"match_phrase_prefix": {"headers": {"query": lexical_keywords, "analyzer": "with_stemmer", "boost": 3}}},
                    {"match": {"content": {"query": lexical_keywords, "boost": 2}}},
                    {"match": {"content": {"query": lexical_keywords, "boost": 1, "fuzziness": "AUTO:4,6"}}},
                    {"wildcard": {"fncname": {"value": f"*{lexical_keywords}*", "boost": 8}}}
                ]
            }
        },
        "size": k
    }
    
    # Execute both queries
    vector_results = es_client.search(index=INDEX_NAME, body=vector_query)
    lexical_results = es_client.search(index=INDEX_NAME, body=lexical_query)
    
    # Combine results using RRF (Reciprocal Rank Fusion)
    combined_docs = {}
    
    # Add vector results with higher weight
    for i, hit in enumerate(vector_results['hits']['hits']):
        doc_id = hit['_id']
        rrf_score = 1 / (30 + i + 1)
        combined_docs[doc_id] = {'doc': hit['_source'], 'score': rrf_score * 1.2}
    
    # Add lexical results
    for i, hit in enumerate(lexical_results['hits']['hits']):
        doc_id = hit['_id']
        rrf_score = 1 / (30 + i + 1)
        if doc_id in combined_docs:
            combined_docs[doc_id]['score'] += rrf_score
        else:
            combined_docs[doc_id] = {'doc': hit['_source'], 'score': rrf_score}
    
    # Sort by combined score and return top k
    sorted_docs = sorted(combined_docs.values(), key=lambda x: x['score'], reverse=True)[:k]
    return [item['doc'] for item in sorted_docs]

# --- 3. Build the RAG Pipeline ---

# Define the RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for Appian developers. Use the provided context from Appian documentation along with your knowledge to give comprehensive answers.

If the context contains relevant information, prioritize it. If the context doesn't fully answer the question, supplement with your general knowledge about Appian. 

Context from Appian Documentation:
{context}

Question: {question}

Answer:
""")

# Create the RAG chain with hybrid search
def format_docs(docs):
    return "\n\n".join(doc.get('content', '') for doc in docs)

def add_sources(response_and_docs):
    response = response_and_docs["response"]
    docs = response_and_docs["docs"]
    
    # Extract paths and titles in relevance order (no deduplication to preserve order)
    sources = []
    seen_paths = set()
    for doc in docs:
        path = doc.get('path', '')
        title = doc.get('title', '')
        if path and path not in seen_paths:
            sources.append({'path': path, 'title': title})
            seen_paths.add(path)
    
    # Format sources in relevance order with titles as link text
    if sources:
        sources_text = "\n\nSources:\n"
        for source in sources:
            title = source['title'] if source['title'] else source['path']
            sources_text += f"- [{title}](https://docs.appian.com/suite/help/25.3/{source['path']})\n"
        return response + sources_text
    return response

def vector_only_search(query: str, k: int = 6) -> List[Dict]:
    """Fallback to vector-only search"""
    query_embedding = ollama_embeddings.embed_query(query)
    
    vector_query = {
        "knn": {
            "field": "vector",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": k * 5,
            "filter": {
                "bool": {
                    "must_not": [
                        {"match_phrase": {"title": {"query": "Deprecated"}}},
                        {"match": {"hideinsearch": True}},
                        {"match": {"islatestsubprodver": False}}
                    ]
                }
            }
        }
    }
    
    results = es_client.search(index=INDEX_NAME, body=vector_query)
    return [hit['_source'] for hit in results['hits']['hits']]

def hybrid_rag_chain(question: str) -> str:
    # Try hybrid search first
    docs = hybrid_search(question)
    
    # If hybrid search returns few results, fallback to vector-only
    if len(docs) < 3:
        docs = vector_only_search(question)
    
    # Format context
    context = format_docs(docs)
    
    # Generate response using proper message format for Bedrock
    formatted_prompt = rag_prompt.format(context=context, question=question)
    response_message = bedrock_llm.invoke([HumanMessage(content=formatted_prompt)])
    response = response_message.content
    
    # Add sources
    return add_sources({"response": response, "docs": docs})

# --- 4. Gradio Interface ---

def chat_interface(message, history):
    """Gradio chat interface function"""
    try:
        response = hybrid_rag_chain(message)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="📚 Appian Documentation Assistant",
    description="Ask questions about Appian and get answers from the official documentation.",
    examples=[
        "How do I create a process model?",
        "What are the different types of Appian objects?",
        "How do I configure user authentication?",
        "What is the difference between a!forEach and a!map?"
    ]
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False
    )