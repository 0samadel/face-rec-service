# Use a modern, slim base image. Python 3.9 is great for these packages.
FROM python:3.9-slim-bullseye

WORKDIR /app

# We do NOT need cmake or build-essential anymore!
# This makes the build much faster.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x ./start.sh

EXPOSE 10000
CMD ["./start.sh"]
