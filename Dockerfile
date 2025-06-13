# Use the proven stable combination: Python 3.8 on Debian Bullseye
FROM python:3.8-bullseye

WORKDIR /app

# Install all necessary system build tools from the modern Bullseye repository
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

# --- The Final Fix: Set CMake Environment Variable ---
# This variable forces the dlib build script to use modern CMake policies,
# which resolves the confusing "< 3.5" error.
ENV CMAKE_ARGS="-DCMAKE_POLICY_DEFAULT_CMP0074=NEW"

# Install the Python dependencies from your requirements.txt
# dlib will now compile correctly using the environment variable above.
RUN pip install --no-cache-dir -r requirements.txt

# --- End of Fix ---

# Copy the rest of your application code
COPY . .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose port and define start command
EXPOSE 10000
CMD ["./start.sh"]
