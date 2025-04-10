FROM python:3.9-slim

WORKDIR /app

COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install nest-asyncio

COPY deployment ./deployment
COPY models ./models

ENV PYTHONPATH=/app

EXPOSE 8502

CMD ["streamlit", "run", "deployment/app.py", "--server.port=8502", "--server.address=0.0.0.0"]