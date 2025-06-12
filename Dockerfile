# Use the full, non-slim version of Python 3.9 on Debian Bullseye.
# Bullseye (Debian 11) is a very stable and common choice for this kind of work.
# This image is larger but contains more build utilities out of the box.
FROM python:3.9-bullseye AS builder

# Set the working directory
WORKDIR /app

# Update package list and install build dependencies.
# We are adding 'pkg-config' which helps cmake find libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Production Stage ---
# Start from a clean, smaller image for the final product
FROM python:3.9-slim-bullseye

WORKDIR /app

# Copy the pre-installed python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Copy your application code
COPY . .

# Make the start.sh script executable
RUN chmod +x ./start.sh

# Expose the port
EXPOSE 10000

# Set the start command
CMD ["./start.sh"]
