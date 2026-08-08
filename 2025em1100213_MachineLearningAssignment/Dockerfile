# Mini Production ML System - Serving API container
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal on purpose (no compilers needed for these wheels).
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://packagefeedproxy.microsoft.io/pypi/simple/ \
    --trusted-host packagefeedproxy.microsoft.io \
    -r requirements.txt

# Copy only what the serving path needs to run + the pieces it imports.
COPY src/ ./src/
COPY serving/ ./serving/
COPY configs/ ./configs/
COPY models/ ./models/

EXPOSE 8000

# NOTE: this image expects models/current_model.joblib and models/scaler.joblib
# to already exist (produced by `python scripts/train.py`) since training is
# treated as a separate offline job, not something that runs at container
# startup. Mount/rebuild with fresh artifacts after each promoted training run.
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
