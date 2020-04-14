# Install Nvidia-Docker from https://github.com/NVIDIA/nvidia-docker
# sudo docker build -t voice-clone-service:v1 .
# sudo docker run -it --gpus=all -p 5000:5000 -v $PWD/wavs:/wavs voice-clone-service:v1
# Puts wavs in wavs folder
# Call API to generate
# curl --location --request POST 'localhost:5000/clone_voices' --header 'Content-Type: application/json' --header 'Content-Type: text/plain' --data-raw '{"voiceFile": "/wavs/weapon.wav", "messages": [""Oh potatoes and molasses If you want some, oh just ask us. They'\''re warm and soft like puppies and socks. Filled with cream and candy rocks."]}'
# Then ask for the last generated wav
# curl --location --request GET 'localhost:5000/lastGeneratedWav'

FROM nvidia/cuda:9.2-base-ubuntu16.04
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         python-qt4 \
         libjpeg-dev \
         zip \
         unzip \
         ffmpeg \
         libportaudio2 \
         wget \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTHON_VERSION=3.6

COPY . /voice-clone
WORKDIR /voice-clone
RUN wget https://williamkarnavas.com/voiceTransfer/pretrained.zip && unzip pretrained.zip

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build

ENV PATH=$PATH:/opt/conda/bin/
ENV USER fastai
# Create Enviroment
COPY enviroment.yaml /enviroment.yaml
RUN conda env create -f enviroment.yaml

# Activate Source
CMD source activate voice-clone-venv
CMD source ~/.bashrc


WORKDIR /voice-clone

ENV FLASK_APP voice_cloning_server.py

# FLASK API PORT
EXPOSE 5000

CMD ["/voice-clone/run.sh"]
