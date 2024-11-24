# chatbot_api/Dockerfile

FROM python:3.11-slim

WORKDIR /app
COPY ./my-fastapi-project/src/ /app

COPY ./my-fastapi-project/src/pyproject.toml /code/pyproject.toml
RUN pip install /code/.

EXPOSE 8000
CMD ["sh", "entrypoint.sh"]
