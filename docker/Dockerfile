FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY examples/ ./examples/

# hatch-vcs needs a real git checkout to derive a version; this image only gets
# a copy of the source tree, so give it a placeholder version instead.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0

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
