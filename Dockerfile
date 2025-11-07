# Use PyTorch official image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir matplotlib

# Copy project files
COPY scripts/ ./scripts/
COPY .gitignore ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p data models results plots json

# Set Python path
ENV PYTHONPATH=/app/scripts:$PYTHONPATH

# Default command (can be overridden)
CMD ["python", "scripts/train.py", "--help"]

