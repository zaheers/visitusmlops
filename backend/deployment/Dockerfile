FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
# Copy project files
COPY --chown=user . $HOME/app

EXPOSE 7860

# Run Uvicorn - Ensure 'main' matches your filename (main.py) and 'app' matches your FastAPI object
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
