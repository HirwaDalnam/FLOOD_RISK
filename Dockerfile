FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + exported model artifacts (model, scalers, config, seed data)
COPY app.py .
COPY lstm_model.keras .
COPY feature_scaler.pkl .
COPY target_scaler.pkl .
COPY config.json .
COPY seed_data.csv .

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
