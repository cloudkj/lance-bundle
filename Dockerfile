FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
COPY examples/ ./examples/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    jq \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[export,test]"

# Additional packages for testing and interacting with integrations
RUN pip install --no-cache-dir \
                beautifulsoup4 \
                brotli \
                datasets \
                docling \
                langchain_community \
                langchain_chroma \
                llama_index \
                llama-index-embeddings-huggingface \
                ollama

# Swap out for headless OpenCV package
RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless

CMD ["bash"]
