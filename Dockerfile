FROM python:3.11.10-slim

ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
