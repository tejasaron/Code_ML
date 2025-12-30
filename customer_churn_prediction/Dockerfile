FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy project files
COPY src ./src
COPY artifacts ./artifacts

# Expose API port
EXPOSE 8000

#Run FastAPI app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


