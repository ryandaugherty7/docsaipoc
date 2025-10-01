FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*
    
# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Pre-download the embedding model during build
RUN ollama serve & sleep 5 && ollama pull mxbai-embed-large && pkill ollama

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY docs-ai-chat.py .

# Expose port 8080 (App Runner standard)
EXPOSE 443

# Health check
HEALTHCHECK CMD curl --fail http://localhost:443/_stcore/health

# Start Ollama and Streamlit with WebSocket-friendly settings
CMD ["sh", "-c", "ollama serve & sleep 5 && streamlit run docs-ai-chat.py --server.port=443 --server.address=0.0.0.0 --server.enableWebsocketCompression=false --server.enableCORS=false --server.allowRunOnSave=false --server.headless=true --server.runOnSave=false -- --web"]

