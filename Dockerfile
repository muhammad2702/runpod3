# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (for Flask or FastAPI)
EXPOSE 8000

# Command to run the application
CMD ["python", "final.py"]

