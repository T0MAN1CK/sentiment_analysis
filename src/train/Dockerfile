# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy train scripts
COPY src/train/ /app/train/

# Ensure required directories exist
RUN mkdir -p /app/data /app/outputs /app/logs

# Install dependencies
RUN pip install --no-cache-dir -r /app/train/requirements.txt

# Download NLTK Resources (Including punkt_tab)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Run scripts in the required order
CMD ["bash", "-c", "python /app/train/data_loader.py && python /app/train/text_processing.py && python /app/train/vectorization.py && python /app/train/train.py"]
