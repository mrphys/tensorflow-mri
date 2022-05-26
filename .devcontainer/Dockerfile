FROM ghcr.io/mrphys/tensorflow-manylinux:1.10.0

RUN apt-get install wget

# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm /etc/apt/sources.list.d/cuda*.list

# To enable plotting.
RUN apt-get update && \
    apt-get install -y libcairo2-dev libgirepository1.0-dev libgtk-3-dev

ARG PYBIN=/usr/local/bin/python
ARG PYVERSIONS="3.7 3.8 3.9 3.10"
RUN for PYVER in ${PYVERSIONS}; do ${PYBIN}${PYVER} -m pip install pycairo PyGObject; done

# For Jupyter notebooks.
RUN for PYVER in ${PYVERSIONS}; do ${PYBIN}${PYVER} -m pip install ipykernel; done

# Install TFMRI dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN for PYVER in ${PYVERSIONS}; do ${PYBIN}${PYVER} -m pip install -r /tmp/requirements.txt; done

# Create non-root user.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add user to sudoers.
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Change default shell to bash.
    usermod --shell /bin/bash $USERNAME
