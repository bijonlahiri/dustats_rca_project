FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first for layer caching
# COPY pyproject.toml uv.lock ./

# Install dependencies (no project, just deps)
# RUN uv sync --frozen --no-install-project --no-dev && rm -rf ~/.cache/uv

# Copy application source
COPY . .

# Install the project itself
RUN uv sync --frozen

#Expose port
EXPOSE 8000

# CMD ["uv", "run", "fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]