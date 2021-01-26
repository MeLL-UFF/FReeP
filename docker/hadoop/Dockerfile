FROM cloudera/quickstart:latest

RUN echo "export JAVA_HOME=/usr/java/jdk1.7.0_67-cloudera" >> /root/.bashrc
RUN echo "export HADOOP_HOME=/usr/lib/hadoop" >> /root/.bashrc
# pra conseguir instalar o pydoop
RUN yum install -y gcc g++ wget zlib-devel bzip2-devel sqlite sqlite-devel openssl-devel \
                    python-devel yum-utils centos-release-scl openldap-devel && \
    yum -y --enablerepo=centos-sclo-rh-testing install devtoolset-7-gcc devtoolset-7-gcc-c++ libsasl2-dev

WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.5.4/Python-3.5.4.tgz
RUN tar xzf Python-3.5.4.tgz
WORKDIR Python-3.5.4
RUN ./configure && make altinstall
RUN rm /opt/Python-3.5.4.tgz
RUN echo "alias python='/usr/local/bin/python3.5'" >> /root/.bashrc

RUN mkdir -p /app
WORKDIR /app
COPY . /app
COPY libs/custom_quickstart.sh /usr/bin
RUN chmod +x /usr/bin/custom_quickstart.sh
RUN python3.5 /app/libs/get-pip.py
RUN source /opt/rh/devtoolset-7/enable && pip install -r /app/requirements.txt