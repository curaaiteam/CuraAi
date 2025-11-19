# =====================================================
# Cura AI Dockerfile - Optimized for GGUF
# =====================================================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for llama-cpp-python
# --no-install-recommends keeps the image smaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .
COPY vector.py .

# Expose the application port
EXPOSE 7860

# Define environment variable to ensure output is not buffered
ENV PYTHONUNBUFFERED=1

# Start the FastAPI app using uvicorn, reading the port from the environment variable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]