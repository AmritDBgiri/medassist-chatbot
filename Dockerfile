FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data and warm up imports to speed up startup
RUN python -c "import langchain; import faiss; print('imports OK')"

# Copy app
COPY . .

# Create uploads dir
RUN mkdir -p uploads

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
