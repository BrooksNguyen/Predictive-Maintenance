FROM python:3.11-slim

WORKDIR /app
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models are mounted via docker-compose volume at /app/models
CMD ["python", "-u", "inference.py"]
