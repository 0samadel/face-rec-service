# Use a modern, slim base image.
FROM python:3.9-slim-bullseye

WORKDIR /app

# Install the missing system library required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose port and define start command
EXPOSE 10000
CMD ["./start.sh"]
