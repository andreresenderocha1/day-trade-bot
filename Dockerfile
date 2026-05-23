FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot_kraken.py .

ENV DATA_DIR=/data

CMD ["python", "-u", "bot_kraken.py"]
