FROM python:3.12

COPY ./requirements.txt ./src/requirements.txt
COPY ./src/ ./src/

RUN pip install -r ./src/requirements.txt

