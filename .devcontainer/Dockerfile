FROM ghcr.io/mrphys/tensorflow-manylinux:1.10.0

# To enable plotting.
RUN apt-get update && \
    apt-get install -y libcairo2-dev libgirepository1.0-dev libgtk-3-dev

ARG PYBIN=/usr/local/bin/python
ARG PYVERSIONS="3.7 3.8 3.9 3.10"
RUN for PYVER in ${PYVERSIONS}; do ${PYBIN}${PYVER} -m pip install pycairo PyGObject; done

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
