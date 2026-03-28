FROM python:3.11-slim

# HuggingFace Spaces requirement: run as user 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Switch to non-root user (required by HF Spaces)
USER user

# HuggingFace Spaces ALWAYS uses port 7860
EXPOSE 7860

# Start server - use python -m to avoid path issues
CMD ["python", "api/server.py"]
