FROM python:3.11-slim

WORKDIR /app
COPY ./src/ /app

RUN chmod +x entrypoint.sh
RUN pip install -r requirements.txt

CMD ["sh", "entrypoint.sh"]
