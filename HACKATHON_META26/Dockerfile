FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env.py .
COPY app.py .
COPY tasks.py .
COPY inference.py .
COPY openenv.yaml .
COPY disaster_ppo_final.zip .

EXPOSE 7860

CMD ["python", "app.py"]
