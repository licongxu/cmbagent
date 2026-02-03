# CMBAgent Backend Dockerfile
#
# This containerizes the FastAPI backend that orchestrates AI agents.
# Code execution happens on the frontend (user's machine), not in this container.

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
# LaTeX is required for Denario paper mode (PDF compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-bibtex-extra \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire cmbagent package (needed for imports)
COPY pyproject.toml README.md ./
COPY cmbagent/ ./cmbagent/

# Install cmbagent package with server dependencies
RUN pip install -e ".[server]"

# Install Denario from source (optional - enables research pipeline modes)
# Copy denario/ into build context before building:
#   cp -r ../Denario /path/to/cmbagent/denario-pkg
# Or clone it:
#   git clone https://github.com/CMBAgents/Denario.git denario-pkg
COPY denario-pkg/ ./denario-pkg/
RUN pip install ./denario-pkg/ || echo "WARNING: Denario not available, research pipeline modes disabled"

# Copy backend code
COPY backend/*.py ./backend/

# Set working directory to backend
WORKDIR /app/backend

# Create directory for credentials (mounted as volume)
RUN mkdir -p /app/credentials

# Create work directory
RUN mkdir -p /app/workdir

# Environment variables (can be overridden at runtime)
ENV CMBAGENT_WORK_DIR=/app/workdir \
    FIREBASE_CREDENTIALS=/app/credentials/firebase-credentials.json \
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/firebase-credentials.json

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run with uvicorn
# Workers can be adjusted via UVICORN_WORKERS env var
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-4}"]
