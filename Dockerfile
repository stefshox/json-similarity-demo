FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV JSON_FILE=apps
CMD ["fastapi", "run", "similarity_api.py"]
