# reference: https://hub.docker.com/_/ubuntu/
FROM continuumio/anaconda3:latest
# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="Daniel Junior <danieljunior@id.uff.br>"
# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Install Ubuntu packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    curl \
    git-core \
    htop \
    pkg-config \
    unzip \
    tree \
    freetds-dev
# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# Update Anaconda
RUN conda update conda && conda update anaconda && conda update --all
RUN pip install -e git+https://github.com/joeddav/get_smarties.git#egg=get_smarties
# Create empty directory to attach volume
RUN mkdir ~/FReeP
ADD . ~/FReeP
WORKDIR ~/FReeP
