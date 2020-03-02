FROM continuumio/anaconda3:anaconda3-5.0.0
LABEL author="danieljunior" email="danieljunior@id.uff.br"

RUN mkdir -p /app
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip & \
    pip install -e git+https://github.com/joeddav/get_smarties.git#egg=get_smarties