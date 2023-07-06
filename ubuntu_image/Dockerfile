# syntax=docker/dockerfile:1
FROM ubuntu:20.04

# set working directory to home
WORKDIR /home

# install python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# install git
RUN apt-get install -y git

# install fortran compiler and build essentials (auto-07p requirements)
RUN apt-get install -y gfortran build-essential

# install auto-07p
RUN git clone https://github.com/auto-07p/auto-07p
WORKDIR /home/auto-07p
RUN /bin/bash ./configure
RUN make
RUN python3 setup.py install
WORKDIR /home

# install rectipy with CPU-only torch build
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install rectipy

# install pycobi
RUN pip3 install pycobi

# install jupyter
RUN pip3 install notebook

# get the use examples repository
RUN git clone https://github.com/pyrates-neuroscience/use_examples
WORKDIR /home/use_examples
