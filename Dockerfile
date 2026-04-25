FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

EXPOSE 7860

# OpenEnv convention: run the server module
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]