FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    htop \
    byobu \
    xserver-xorg-dev \
    patchelf \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install conda
ENV LANG C.UTF-8
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Install Mujoco-py
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
#    && python3.8 get-pip.py
#RUN git clone https://github.com/openai/mujoco-py \
#    && cd mujoco-py/ && pip install --no-cache-dir -r requirements.txt \
#    && pip install --no-cache-dir -r requirements.dev.txt \
#    && python3 setup.py install
#
#RUN pip install torch matplotlib gym tqdm tabulate gym

