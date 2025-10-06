#!/usr/bin/env python3

import os
import sys
import re
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_aws import BedrockEmbeddings

# Configuration
INDEX_NAME = "appian_docs_rag"
ELASTICSEARCH_URL = os.getenv('SOURCE_ES_HOST', 'https://29944f54da01413ab55da3b9f2fa68ad.us-east-1.aws.found.io:443')
ES_API_KEY = os.getenv('SOURCE_ES_API_KEY')

# Initialize Bedrock embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

def chunk_content(content, max_chars=8192):
    """Chunk content by character limit, breaking at sentence boundaries"""
    if len(content) <= max_chars:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + max_chars
        
        if end >= len(content):
            chunks.append(content[start:])
            break
        
        # Find the nearest sentence end before the limit
        chunk_text = content[start:end]
        sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', chunk_text)]
        
        if sentence_ends:
            # Use the last sentence end within the limit
            actual_end = start + sentence_ends[-1]
            chunks.append(content[start:actual_end])
            start = actual_end
        else:
            # No sentence end found, just cut at the limit
            chunks.append(content[start:end])
            start = end
    
    return chunks

def main():
    # Get source index from command line argument
    try:
        load_index = sys.argv.index('-load')
        if load_index + 1 < len(sys.argv):
            SOURCE_INDEX = sys.argv[load_index + 1]
        else:
            print("Error: -load requires a source index name")
            print("Usage: load_data.py -load <source_index>")
            exit()
    except ValueError:
        print("Error: -load parameter not found")
        print("Usage: load_data.py -load <source_index>")
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
    
    # Index documents with chunking and embeddings
    documents = []
    total_chunks = 0
    
    for hit in json_data['hits']['hits']:
        source = hit['_source']
        content = source.get('content', '').strip()
        
        # Skip documents with empty content
        if not content:
            continue
        
        # Chunk the content
        chunks = chunk_content(content)
        
        for chunk in chunks:
            # Generate embedding for chunk
            embedding = bedrock_embeddings.embed_query(chunk)
            
            # Prepare document with all fields (chunk in content field)
            doc = {
                "vector": embedding,
                "content": chunk,
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
            total_chunks += 1
    
    # Bulk index documents
    actions = [
        {
            "_index": INDEX_NAME,
            "_source": doc
        }
        for doc in documents
    ]
    
    bulk(es_client, actions)
    print(f"Successfully indexed {total_chunks} chunks from {len(json_data['hits']['hits'])} source documents.")

if __name__ == "__main__":
    main()