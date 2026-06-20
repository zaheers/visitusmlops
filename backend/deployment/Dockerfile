# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to optimize Docker layer caching
COPY requirements.txt .

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security (Required for Hugging Face Spaces / secure cloud deployment)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch to the new user's home directory
WORKDIR $HOME/app

# Copy the API script and the .joblib model file with the correct user permissions
COPY --chown=user . $HOME/app

# Expose the standard FastAPI port
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
