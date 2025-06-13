# We can use the slim image because we are no longer compiling.
FROM python:3.8-slim-bullseye

WORKDIR /app

# We NO LONGER need apt-get install for cmake and build-essential.
# This makes the build much faster and the image smaller.

# Copy requirements file.
COPY requirements.txt .

# Install the Python dependencies. Pip will download the pre-compiled wheel.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose port and set start command
EXPOSE 10000
CMD ["./start.sh"]
