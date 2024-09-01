FROM jupyter/minimal-notebook

COPY ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt