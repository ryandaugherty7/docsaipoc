#!/usr/bin/env python3

import os
import sys
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
INDEX_NAME = "docs-index"
ELASTICSEARCH_URL = os.getenv('SOURCE_ES_HOST', 'https://29944f54da01413ab55da3b9f2fa68ad.us-east-1.aws.found.io:443')
ES_API_KEY = os.getenv('SOURCE_ES_API_KEY')

# Initialize embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

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
    
    # Index documents with both vector embeddings and text fields
    documents = []
    for hit in json_data['hits']['hits']:
        source = hit['_source']
        content = source.get('content', '').strip()
        
        # Skip documents with empty content
        if not content:
            continue
            
        # Generate embedding for content
        embedding = hf_embeddings.embed_query(content)
        
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
    actions = [
        {
            "_index": INDEX_NAME,
            "_source": doc
        }
        for doc in documents
    ]
    
    bulk(es_client, actions)
    print(f"Successfully indexed {len(documents)} documents with hybrid search capabilities.")

if __name__ == "__main__":
    main()