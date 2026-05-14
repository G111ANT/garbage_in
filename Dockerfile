FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN git clone  /app

ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --locked

EXPOSE 8080
CMD ["uv", "run", "main.py"]