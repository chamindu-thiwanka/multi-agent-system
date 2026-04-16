# ================================================================
# backend/Dockerfile
# ================================================================
# Instructions for building the backend Docker container.
#
# HOW DOCKER BUILDS AN IMAGE:
# Each line is a "layer". Docker caches layers and only rebuilds
# changed layers. This makes rebuilds fast.
#
# LAYER ORDER MATTERS:
# Put things that change rarely FIRST (installs, dependencies).
# Put things that change often LAST (your actual code).
# This way, code changes only rebuild the last few layers,
# not the entire dependencies layer.
# ================================================================

# Start from official Python 3.11 slim image
# "slim" = minimal base image (fewer pre-installed packages)
# Smaller = faster to download, less disk space, fewer security risks
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands run from /app
# Our code will be at /app/main.py, /app/config.py, etc.
WORKDIR /app

# Install system dependencies first
# These change rarely, so this layer is cached
# 
# curl: needed for health checks in docker-compose.yml
# build-essential: needed for compiling some Python packages
# --no-install-recommends: skip optional packages (smaller image)
# rm -rf /var/lib/apt/lists/*: clean apt cache (smaller image)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt BEFORE copying code
# WHY? If we copy all code first, any code change invalidates
# the pip install cache layer. By copying requirements.txt alone,
# pip install only reruns when requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't save pip's download cache (smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
# This layer changes every time you edit any backend file
COPY . .

# Expose port 8000 inside the container
# This is documentation - the actual port mapping is in docker-compose.yml
EXPOSE 8000

# The command to run when the container starts
# uvicorn main:app: run the 'app' object from 'main.py'
# --host 0.0.0.0: accept connections from any IP (required for Docker)
# --port 8000: listen on port 8000
# --workers 1: one worker process (fine for development)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]