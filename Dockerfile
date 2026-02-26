FROM python:3.10-slim

WORKDIR /app

# Switch apt to Aliyun mirror for faster downloads in China
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# Install ffmpeg and git for audio processing and submodule cloning
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies (using Tsinghua pip mirror)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

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
