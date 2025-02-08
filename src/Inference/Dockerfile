# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy inference scripts
COPY src/Inference/ /app/inference/

# Ensure required directories exist
RUN mkdir -p /app/outputs /app/logs

# Install dependencies
RUN pip install --no-cache-dir -r /app/inference/requirements.txt

# Set default command to run inference (chooses best model automatically)
CMD ["python", "/app/inference/run_inference.py"]
