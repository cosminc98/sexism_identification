FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

ARG VENV_PATH=/opt/dev-venv

ENV VENV_ACTIVATE=${VENV_PATH}/bin/activate

RUN apt update
RUN apt install -y python3.10-venv vim

# free up some space
RUN apt-get autoclean && apt-get clean

RUN python3.10 -m venv ${VENV_PATH}
COPY requirements.txt .
RUN . ${VENV_ACTIVATE} && \
    pip install Flake8-pyproject black pylint isort && \
    pip install -r requirements.txt
RUN echo "source ${VENV_ACTIVATE}" >> ~/.bashrc
