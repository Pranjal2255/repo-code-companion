# Use an optimized, official lightweight Python runtime
FROM python:3.11-slim

# Set system environment variables to optimize Python inside the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container image filesystem
WORKDIR /app

# Install system dependencies required for Git cloning and compiling components
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker's caching layers
COPY requirements.txt .

# Upgrade pip inside the container to handle modern wheel hashes cleanly
RUN pip install --no-cache-dir --upgrade pip

# Install the dependencies directly into the container system space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your local application code into the container filesystem
COPY . .

# Expose the default networking port that Streamlit listens on
EXPOSE 8501

# Run the Streamlit web server when the container starts up
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]