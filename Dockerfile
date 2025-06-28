FROM python:3.11-slim

# Add required system dependencies
RUN apt-get update && apt-get install -y \
    libexpat1 \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for rasterio if needed
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set workdir and install python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
