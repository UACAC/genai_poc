# Use an official Python image
FROM python:3.11.9-slim

WORKDIR /app

# (Optional) Install any system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ChromaDB plus server extras + the tools you need (FastAPI, Uvicorn)
RUN pip install --no-cache-dir chromadb[server] fastapi uvicorn

# Copy only the chromadb subfolder into /app
COPY . /app

# Expose port 8000 for our custom server
EXPOSE 8020

# Start our custom server script
CMD ["python", "chromadb_main.py"]
