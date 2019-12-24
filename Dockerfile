FROM python:3.6-slim

COPY requirements.txt /root
WORKDIR /root
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN pip install sqlalchemy psycopg2-binary
RUN pip install flask_sqlalchemy
RUN pip install pandas

