FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# CPU wheels avoid shipping unnecessary CUDA libraries.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py ./
COPY static ./static
COPY model ./model
# Fail clearly when a hosting service checks out an LFS pointer instead of weights.
RUN python -c "from pathlib import Path; p=Path('model/epoch61.pt'); assert p.stat().st_size > 1000000, 'Download Git LFS model weights before building'"
RUN mkdir -p /data
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn server:combined_app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
