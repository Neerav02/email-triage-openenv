FROM python:3.11-slim

# Create user with uid 1000 (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Fix permissions
RUN chown -R user:user /app

# Switch to non-root user
USER user

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Run the server
CMD ["python", "api/server.py"]
