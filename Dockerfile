FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first for layer caching
# COPY pyproject.toml uv.lock ./

# Install dependencies (no project, just deps)
# RUN uv sync --frozen --no-install-project --no-dev && rm -rf ~/.cache/uv

# --- Lambda Web Adapter additions (the only changes vs. the video's Dockerfile) ---
# Drops a Lambda extension binary into /opt/extensions. The binary is inert
# unless invoked by the Lambda runtime, so local `docker run` is unaffected.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Tell the adapter which port FastAPI listens on
ENV PORT=8000

# Async init: the adapter waits for /api/health to return 200 before
# forwarding the first request. This prevents 502s during slow cold starts
# where the MLflow model and LangGraph are still loading.
ENV AWS_LWA_ASYNC_INIT=true
ENV AWS_LWA_READINESS_CHECK_PATH=/api/health
ENV AWS_LWA_READINESS_CHECK_PROTOCOL=http

# Change uv cache dir to support lambda /tmp
ENV UV_CACHE_DIR=/tmp/uv-cache

# Conversation history to S3
ENV USE_S3=true
ENV S3_BUCKET=dustats-rca-conversation-history

# Copy application source
COPY . .

# Install the project itself
RUN uv sync --frozen --extra cpu

#Expose port
EXPOSE 8000

# CMD ["uv", "run", "fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]