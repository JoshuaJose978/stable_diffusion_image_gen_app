# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV TORCH_HOME=/app/cache
ENV HF_HOME=/app/cache
ENV STREAMLIT_SERVER_PORT=8501

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
