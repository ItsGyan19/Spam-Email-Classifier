FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5001

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 5001

# Launch production server with Gunicorn
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app.app:app

