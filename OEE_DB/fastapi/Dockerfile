FROM python:3.10-slim

WORKDIR /app

RUN pip install fastapi uvicorn sqlalchemy psycopg2-binary scikit-learn joblib numpy

COPY . /app

# (3) Expose-Port 
EXPOSE 8000

# (4) Start-Befehl für FastAPI/Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
