FROM python:3.8-slim

WORKDIR /app

# install build deps for xgboost wheel etc
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install requirements first so the wheel layer is cached
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the model + scaler pkls live in /app/models and are mounted in or COPY'd
# from a built artifact. The image does not retrain on container start.
ENV PORT=5000 THRESHOLD=0.5
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:'+'${PORT:-5000}'+'/health').getcode()==200 else sys.exit(1)" || exit 1

# heroku-friendly: respect $PORT
CMD gunicorn -b 0.0.0.0:${PORT} app:app --workers 2 --timeout 60
