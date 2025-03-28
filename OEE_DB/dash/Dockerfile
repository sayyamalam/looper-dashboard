FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir dash==2.11.1 \
                         dash-bootstrap-components==1.3.0 \
                         plotly==5.15.0 \
                         requests==2.31.0

COPY . /app

EXPOSE 8050

CMD ["python", "app.py"]
