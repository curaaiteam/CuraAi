# =====================================================
# CuraAi - Hugging Face Spaces Docker
# =====================================================

FROM python:3.10-slim

WORKDIR /app

# Copy application files
COPY requirements.txt main.py vector.py ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

#reload trigger