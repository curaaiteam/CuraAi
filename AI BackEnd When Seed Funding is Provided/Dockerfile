# =====================================================
# Cura AI Dockerfile
# =====================================================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port defined in .env (7860)
EXPOSE 7860

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
