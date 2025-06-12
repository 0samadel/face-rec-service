# ------------------------------------------------------------------
# Stage 1: Build Stage - Install dependencies and compile dlib
# ------------------------------------------------------------------
# Use a specific version of Debian/Ubuntu that has the required build tools
FROM python:3.9-slim-buster AS builder

# Set the working directory
WORKDIR /app

# Update package lists and install system dependencies required for dlib and opencv
# cmake is the most important one here
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# Install Python packages. This is where dlib will be compiled.
# Using --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# Stage 2: Final Stage - Create the production image
# ------------------------------------------------------------------
# Use a slim base image to keep the final container small
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the pre-installed python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Copy your application code
COPY . .

# Expose the port your app will run on
EXPOSE 10000

# The command to run your application using gunicorn
# This will be overridden by Render's Start Command, but it's good practice
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]