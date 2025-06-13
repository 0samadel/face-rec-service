# Use a modern, stable base image
FROM python:3.8-bullseye

WORKDIR /app

# Install all necessary system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in one go
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x ./start.sh

# Expose port and define start command
EXPOSE 10000
CMD ["./start.sh"]
