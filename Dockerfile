# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . ./src/

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV MLFLOW_TRACKING_URI="http://mlflow-server:5000"

# Set the entry point
ENTRYPOINT ["python", "-m", "app.main"]
CMD ["--config", "som-config-file.yml", "--stage", "all"]
