# Use an official lightweight Python image
# Python 3.11 is stable and compatible with all used libraries
FROM python:3.11-slim

# Set environment variables
# PYTHONUNBUFFERED=1 ensures logs are flushed to Cloud Logging immediately
# PYTHONDONTWRITEBYTECODE=1 prevents creation of .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first
# This leverages Docker layer caching: if requirements don't change, 
# Docker won't re-run pip install on subsequent builds.
COPY requirements.txt .

# Install dependencies
# --no-cache-dir keeps the image size small by removing pip cache
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the application source code
# We explicitly copy specific folders to avoid accidental clutter
COPY services/ ./services/
COPY utils/ ./utils/
COPY main.py .

# Expose the port (Cloud Run defaults to 8080)
EXPOSE 8080

# Command to run the application
# We use 'exec' to ensure the process receives signals (like SIGTERM) correctly
# We listen on 0.0.0.0 to allow external access within the container network
# We use the PORT environment variable injected by Cloud Run (defaults to 8080 if missing)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}