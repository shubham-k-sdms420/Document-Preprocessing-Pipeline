#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/uploads /app/outputs

# Ensure .env file exists (should be mounted from host)
# If not mounted, create a default one
if [ ! -f /app/.env ]; then
    echo "# Default .env file" > /app/.env
fi

# Fix permissions for mounted volumes (running as root)
chmod -R 777 /app/uploads /app/outputs 2>/dev/null || true

# Change ownership to appuser (UID 1000) if appuser exists
if id -u appuser > /dev/null 2>&1; then
    chown -R appuser:appuser /app/uploads /app/outputs 2>/dev/null || true
    # Switch to appuser for running the application
    cd /app
    # Properly quote all arguments
    exec su appuser -s /bin/bash -c "exec $(printf '%q ' "$@")"
else
    # If appuser doesn't exist, run as current user (root)
    cd /app
    exec "$@"
fi
