FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1

ARG VENV_PATH=/opt/dev-venv

ENV VENV_ACTIVATE=${VENV_PATH}/bin/activate

RUN apt update
RUN apt install -y python3.10-venv vim

# free up some space
RUN apt-get autoclean && apt-get clean

RUN python3.10 -m venv ${VENV_PATH}
COPY requirements.txt .
RUN source ${VENV_ACTIVATE} && \
    pip install Flake8-pyproject black pylint isort && \
    pip install -r requirements.txt
RUN echo "source ${VENV_ACTIVATE}" >> ~/.bashrc
