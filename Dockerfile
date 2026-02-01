# Multi-stage build for authenticated model downloads
FROM python:3.10-slim AS model-downloader
# Install huggingface-cli
RUN pip install "huggingface_hub[cli]"
# Set working directory
WORKDIR /model-downloader
# Create directory for downloaded models
RUN mkdir -p /model-downloader/models/csm-1b
RUN mkdir -p /model-downloader/models/dia-1.6b

# This will run when building the image
ARG HF_TOKEN
ARG TTS_ENGINE=csm

# Download CSM-1B model (pass token inline, no login step needed)
RUN if [ -n "$HF_TOKEN" ] || [ "$TTS_ENGINE" = "csm" ]; then \
    echo "Downloading CSM-1B model..."; \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sesame/csm-1b', 'ckpt.pt', local_dir='/model-downloader/models/csm-1b', token='${HF_TOKEN}' if '${HF_TOKEN}' else None)"; \
    else echo "Skipping CSM-1B model download"; fi

# Download Dia-1.6B model
RUN if [ -n "$HF_TOKEN" ] || [ "$TTS_ENGINE" = "dia" ]; then \
    echo "Downloading Dia-1.6B model..."; \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('nari-labs/Dia-1.6B', 'config.json', local_dir='/model-downloader/models/dia-1.6b', token='${HF_TOKEN}' if '${HF_TOKEN}' else None)"; \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('nari-labs/Dia-1.6B', 'dia-v0_1.pth', local_dir='/model-downloader/models/dia-1.6b', token='${HF_TOKEN}' if '${HF_TOKEN}' else None)"; \
    else echo "Skipping Dia-1.6B model download"; fi

# Now for the main application stage
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dia.txt .

# Create and set up persistent directories with proper permissions
RUN mkdir -p /app/static /app/models /app/models/csm-1b /app/models/dia-1.6b \
    /app/voice_memories /app/voice_references /app/voice_profiles \
    /app/cloned_voices /app/audio_cache /app/tokenizers /app/logs && \
    chmod -R 777 /app/voice_references /app/voice_profiles /app/voice_memories \
    /app/cloned_voices /app/audio_cache /app/static /app/logs /app/tokenizers /app/models

# Copy static files
COPY ./static /app/static

# Install Python dependencies (CUDA-enabled torch)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install torch torchaudio numpy --index-url https://download.pytorch.org/whl/cu124

# Install torchao from source (needs torch visible at build time)
ENV CUDA_HOME=/usr/local/cuda
RUN pip3 install --no-build-isolation git+https://github.com/pytorch/ao.git

# Install torchtune from source with specific branch for latest features
RUN git clone https://github.com/pytorch/torchtune.git /tmp/torchtune && \
    cd /tmp/torchtune && \
    # Try to use the main branch, which should have llama3_2
    git checkout main && \
    pip install -e .

# Install base requirements
RUN pip3 install -r requirements.txt

# Install Dia model dependencies if TTS_ENGINE is set to dia
ARG TTS_ENGINE=csm
RUN if [ "$TTS_ENGINE" = "dia" ]; then \
    echo "Installing Dia model dependencies..." && \
    pip3 install -r requirements-dia.txt && \
    echo "Dia model dependencies installed"; \
fi

# Install additional dependencies for streaming and voice cloning
RUN pip3 install yt-dlp openai-whisper

# Copy application code
COPY ./app /app/app

# Copy downloaded models from the model-downloader stage
COPY --from=model-downloader /model-downloader/models/csm-1b /app/models/csm-1b
COPY --from=model-downloader /model-downloader/models/dia-1.6b /app/models/dia-1.6b

# Show available models in torchtune
RUN python3 -c "import torchtune.models; print('Available models in torchtune:', dir(torchtune.models))"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3", "-m", "app.main"]