FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Модель обучается при сборке образа: файл всегда совместим с версией scikit-learn внутри контейнера
RUN python -m training.train
CMD ["uvicorn", "api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
