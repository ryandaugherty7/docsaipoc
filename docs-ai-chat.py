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
# Removed agent imports - using simple RAG chain instead
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import List, Dict
import json
from elasticsearch import Elasticsearch
import re
import boto3

# Check if running as Streamlit app
if '--web' in sys.argv:
    import streamlit as st

# --- Environment Setup (Assumes you have Ollama and Elasticsearch running) ---
# Make sure Ollama and Elasticsearch are running on their default ports.
# Ollama: `ollama run llama3`
# Elasticsearch: `docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=true" -e "ELASTIC_PASSWORD=your_password" -e "ELASTIC_USERNAME=elastic" docker.elastic.co/elasticsearch/elasticsearch:8.12.1`

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

# --- 2. Check for -load parameter and conditionally load data ---
load_data = '-load' in sys.argv

if load_data:
    # Get source index from command line argument
    try:
        load_index = sys.argv.index('-load')
        if load_index + 1 < len(sys.argv):
            SOURCE_INDEX = sys.argv[load_index + 1]
        else:
            print("Error: -load requires a source index name")
            print("Usage: docs-ai-chat.py -load <source_index>")
            exit()
    except ValueError:
        print("Error: -load parameter not found")
        exit()
    
    print(f"Loading data from source Elasticsearch index: {SOURCE_INDEX}...")
    
    # Query all documents from source ES
    headers = {
        "Authorization": f"ApiKey {ES_API_KEY}",
        "Content-Type": "application/json"
    }
    
    query = {
        "query": {"match_all": {}},
        "size": 10000  # Adjust size as needed
    }
    
    response = requests.post(
        f"{ELASTICSEARCH_URL}/{SOURCE_INDEX}/_search",
        headers=headers,
        json=query
    )
    
    if response.status_code != 200:
        print(f"Failed to query source Elasticsearch: {response.status_code} - {response.text}")
        exit()
    
    json_data = response.json()
    
    # Create Elasticsearch client
    es_client = Elasticsearch(
        [ELASTICSEARCH_URL],
        api_key=ES_API_KEY,
        verify_certs=True
    )
    
    # Delete existing index if it exists
    try:
        if es_client.indices.exists(index=INDEX_NAME):
            es_client.indices.delete(index=INDEX_NAME)
            print(f"Deleted existing index: {INDEX_NAME}")
    except:
        pass
    
    # Create index with custom settings and mapping
    index_settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "index_analyzer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "filter_stemmer"]
                    },
                    "search_analyzer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "filter_graph_synonyms", "filter_stemmer"]
                    },
                    "with_stemmer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "filter_stemmer"]
                    }
                },
                "filter": {
                    "filter_graph_synonyms": {
                        "type": "synonym_graph",
                        "synonyms_set": "prod-published-25-3-maint",
                        "updateable": True
                    },
                    "filter_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine"
                },
                "content": {
                    "type": "text", 
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer"
                },
                "title": {
                    "type": "text", 
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer"
                },
                "headers": {
                    "type": "text", 
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer"
                },
                "filename": {"type": "text"},
                "path": {"type": "keyword"},
                "hideinsearch": {"type": "boolean"},
                "islatestsubprodver": {"type": "boolean"},
                "reftype": {"type": "keyword"},
                "type": {"type": "keyword"},
                "topic": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "fncname": {"type": "keyword"},
                "searchCategory": {"type": "keyword"}
            }
        }
    }
    
    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    print(f"Created index {INDEX_NAME} with custom settings")
    
    # Index documents with both vector embeddings and text fields
    documents = []
    for hit in json_data['hits']['hits']:
        source = hit['_source']
        content = source.get('content', '').strip()
        
        # Skip documents with empty content
        if not content:
            continue
            
        # Generate embedding for content
        embedding = ollama_embeddings.embed_query(content)
        
        # Prepare document with all fields
        doc = {
            "vector": embedding,
            "content": content,
            "title": source.get('title', ''),
            "headers": source.get('headers', []),
            "filename": source.get('filename', ''),
            "path": source.get('path', ''),
            "hideinsearch": source.get('hideinsearch', False),
            "islatestsubprodver": source.get('islatestsubprodver', True),
            "reftype": source.get('reftype', ''),
            "type": source.get('type', ''),
            "topic": source.get('topic', ''),
            "categories": source.get('categories', []),
            "fncname": source.get('fncname', ''),
            "searchCategory": source.get('searchCategory', [])
        }
        
        documents.append(doc)
    
    # Bulk index documents
    from elasticsearch.helpers import bulk
    actions = [
        {
            "_index": INDEX_NAME,
            "_source": doc
        }
        for doc in documents
    ]
    
    bulk(es_client, actions)
    print(f"Successfully indexed {len(documents)} documents with hybrid search capabilities.")
    
    # Store es_client for later use
    globals()['es_client'] = es_client
else:
    print("Using existing data in Elasticsearch...")
    # Create Elasticsearch client for existing data
    es_client = Elasticsearch(
        [ELASTICSEARCH_URL],
        api_key=ES_API_KEY,
        verify_certs=True
    )
    globals()['es_client'] = es_client

# --- 3. Custom Hybrid Search Implementation ---

def extract_keywords(query: str) -> str:
    """Extract keywords from natural language query for lexical search"""
    # Remove punctuation
    query = re.sub(r'[?!:;.]', '', query)
    # Remove question words and common phrases
    query = re.sub(r'\b(what|are|the|different|how|why|when|where|make|up|that|of|in|to|for|with|by|can|do|does|is|will|would|should|could)\b', '', query, flags=re.IGNORECASE)
    # Keep only meaningful words (3+ chars)
    words = [word.strip() for word in query.split() if len(word.strip()) > 2]
    # print(f"Stripped query: {' '.join(words)}")
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

# --- 4. Build the RAG Pipeline ---

# Define the RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for Appian developers. Use the provided context from Appian documentation along with your knowledge to give comprehensive answers.

If the context contains relevant information, prioritize it. If the context doesn't fully answer the question, supplement with your general knowledge about Appian. 

Context from Appian Documentation:
{context}

Question: {question}

Answer:
""")

# rag_prompt = ChatPromptTemplate.from_template("""
# You are a helpful assistant for Appian developers. Use your general knowledge about Appian and augment it with the provided context from Appian documentation to give comprehensive answers.

# Prioritize your general knowledge about Appian, but always supplement it with the context from Appian documentation, particularly so that sources can be cited. 

# Context from Appian Documentation:
# {context}

# Question: {question}

# Answer:
# """)

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

# --- 5. Interactive Question and Answer Loop ---

if '--web' in sys.argv:
    # Streamlit mode
    st.set_page_config(page_title="Appian Documentation Assistant", page_icon="📚")
    
    @st.cache_resource
    def get_clients():
        return bedrock_llm, ollama_embeddings, es_client
    
    def main():
        st.title("📚 Appian Documentation Assistant")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about Appian..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    response = hybrid_rag_chain(prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    if __name__ == "__main__":
        main()
else:
    # Terminal mode
    print("\n--- Appian Documentation Assistant ---")
    print("Ask questions about Appian. Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() == 'exit':
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...")
            response = hybrid_rag_chain(question)
            print(f"\nAnswer: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")