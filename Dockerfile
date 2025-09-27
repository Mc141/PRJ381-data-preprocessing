FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages required to build some Python packages (rasterio, lxml, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       libpq-dev \
       libxml2-dev \
       libxslt1-dev \
       libgdal-dev \
       gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Make sure pip, wheel, and setuptools are current so binary wheels are used whenever possible
RUN python -m pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start the app with uvicorn (matches Procfile)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
