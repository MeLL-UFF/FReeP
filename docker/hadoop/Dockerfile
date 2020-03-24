FROM sequenceiq/hadoop-docker:2.7.1

RUN echo "export PATH=$PATH:$HADOOP_PREFIX/bin" >> /root/.bashrc
RUN echo "export ARROW_LIBHDFS_DIR='/usr/local/hadoop/lib/native'" >> /root/.bashrc

RUN yum clean all && yum -y update && \
    yum -y install gcc wget zlib-devel bzip2-devel sqlite sqlite-devel openssl-devel

WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.5.4/Python-3.5.4.tgz
RUN tar xzf Python-3.5.4.tgz
WORKDIR Python-3.5.4
RUN ./configure && make altinstall
RUN rm /opt/Python-3.5.4.tgz
RUN echo "alias python='/usr/local/bin/python3.5'" >> /root/.bashrc
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" && python3.5 get-pip.py

RUN mkdir -p /app
WORKDIR /app
COPY . /app
RUN pip install dist/freep-0.0.1-py3-none-any.whl
RUN pip install pyarrow