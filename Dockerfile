# ------------------------------------------------------------------
# Stage 1: Build Stage
# ------------------------------------------------------------------
FROM python:3.9-slim-buster AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# Stage 2: Final Stage
# ------------------------------------------------------------------
FROM python:3.9-slim-buster
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY . .

# --- NEW LINES START HERE ---
# Make the start.sh script executable
RUN chmod +x ./start.sh
# --- NEW LINES END HERE ---

EXPOSE 10000

# The CMD is now optional since Render will use the Start Command, but it's good to have.
CMD ["./start.sh"]