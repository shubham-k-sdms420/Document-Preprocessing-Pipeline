# Multi-stage build for Document Preprocessing Pipeline
FROM python:3.11-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads /app/outputs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Set entrypoint (runs as root to fix permissions, then switches to appuser)
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default to root user (entrypoint will switch to appuser after fixing permissions)
# This allows the entrypoint to fix permissions on mounted volumes
USER root

# Expose Flask port (default 5000, can be overridden via PORT env var)
ARG PORT=5000
EXPOSE ${PORT}

# Health check (uses PORT env var at runtime)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.getenv('PORT', '5000'); urllib.request.urlopen(f'http://localhost:{port}/')" || exit 1

# Run the application
CMD ["python", "app.py"]
