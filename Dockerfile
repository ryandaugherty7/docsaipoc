FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY docs-ai-chat.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start Ollama and Streamlit
CMD ["sh", "-c", "ollama serve & sleep 10 && ollama pull mxbai-embed-large && streamlit run docs-ai-chat.py --server.port=8501 --server.address=0.0.0.0 -- --web"]