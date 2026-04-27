FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p my_pdfs vector_db

EXPOSE 8000

CMD ["bash", "start.sh"]
