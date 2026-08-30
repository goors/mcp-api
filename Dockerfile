FROM python:3.12-slim AS builder

WORKDIR /app

# 1. Install system libs + curl (to download the binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    gcc \
    g++ \
    python3-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# 3. Copy your project files
COPY . .

# 4. Pip install (Nuking the library version is best practice here)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Ensure the app finds the binary in /app/yt-dlp
CMD ["uvicorn", "qwen:app", "--host", "0.0.0.0", "--port", "8000", "--forwarded-allow-ips", "*"]
