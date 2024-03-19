FROM python:3.7-slim-buster
EXPOSE 8501

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["streamlit","run","500_app.py","--server.port=8501","--server.address=0.0.0.0"]