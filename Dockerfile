# Use Python 3.8 on Debian Bullseye (Debian 11). This is a very stable and common combination.
# It provides a newer version of cmake while keeping Python at 3.8.
FROM python:3.8-bullseye

# Set the working directory
WORKDIR /app

# Install build essentials. The versions on Bullseye are newer and more compatible.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose the port and set the start command
EXPOSE 10000
CMD ["./start.sh"]
