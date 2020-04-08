FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install libopencv-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

ENV HOME /home/
WORKDIR /home
