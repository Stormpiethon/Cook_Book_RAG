FROM python:3.9-slim

WORKDIR /app

# Install system deps needed by some packages (grpcio, tensorflow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

ENV FLASK_ENV=production

CMD ["python", "app.py"]
