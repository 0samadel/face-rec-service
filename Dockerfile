# Stick with the stable Python 3.8 on Debian Bullseye
FROM python:3.8-bullseye

WORKDIR /app

# Install system dependencies
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

# --- NEW AND IMPROVED PIP INSTALL ---
# First, upgrade pip, setuptools, and wheel. This is a best practice.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Now, install the requirements using the legacy resolver.
# This flag changes how pip handles dependency conflicts and can solve stubborn issues.
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose the port and set the command
EXPOSE 10000
CMD ["./start.sh"]
