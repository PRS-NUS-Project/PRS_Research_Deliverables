# ===============================
# Base image: Lightweight Python runtime
# ===============================
FROM python:3.12-slim

# ===============================
# Set working directory inside container
# ===============================
WORKDIR /app

# ===============================
# Copy only runtime requirements 
# ===============================
COPY requirements.txt .

# ===============================
# Install Python dependencies for runtime
# ===============================
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy the rest of the project needed at runtime
# ===============================
COPY app ./app
COPY dataset/model_results ./dataset

# ===============================
# Expose Streamlit default port
# ===============================
EXPOSE 8501

# ===============================
# Launch command
# ===============================
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]