# ------------------------------------------------------------------
# Stage 1: Build Stage - Install dependencies and compile dlib
# ------------------------------------------------------------------
# CHANGE THIS LINE: from 'buster' to 'bookworm'
FROM python:3.9-slim-bookworm AS builder

# Set the working directory
WORKDIR /app

# Update package lists and install system dependencies required for dlib and opencv
# The commands stay the same, but they will pull newer versions from Bookworm's repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# Install Python packages.
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# Stage 2: Final Stage - Create the production image
# ------------------------------------------------------------------
# CHANGE THIS LINE TOO: from 'buster' to 'bookworm'
FROM python:3.9-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy the pre-installed python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Copy your application code
COPY . .

# Make the start.sh script executable
RUN chmod +x ./start.sh

# Expose the port your app will run on
EXPOSE 10000

# The command to run your application using gunicorn
CMD ["./start.sh"]
