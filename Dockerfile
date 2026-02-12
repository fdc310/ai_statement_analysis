FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg and git for audio processing and submodule cloning
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Clone Tencent Cloud Speech SDK (git submodule)
RUN git clone https://github.com/TencentCloud/tencentcloud-speech-sdk-python.git \
    app/core/util/tencentcloud-speech-sdk-python

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
