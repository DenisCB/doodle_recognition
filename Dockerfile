FROM python:3.6-slim

COPY requirements.txt /root
WORKDIR /root
RUN pip install -r requirements.txt
