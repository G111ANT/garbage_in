FROM python:3.14-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git
RUN git clone https://github.com/G111ANT/garbage_in.git /app

ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --locked

EXPOSE 8080
CMD ["uv", "run", "main.py"]