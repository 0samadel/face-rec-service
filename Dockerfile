# Start with a known stable Python version for dlib compilation
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Combine all system dependency installation and cleanup in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose the port Render will use
EXPOSE 10000

# Set the command to run your application
CMD ["./start.sh"]
