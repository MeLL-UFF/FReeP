# FReep - Feature Recommender from Preferences


### Build Image

docker build -f docker/original/Dockerfile -t freep .

### Run for development

docker run --rm -v ${PWD}:/app -w /app -it freep bash

### Gerar a lib
- python3 setup.py sdist bdist_wheel

### Criar imagem Hadoop

- docker build -f docker/hadoop/Dockerfile -t freep_hadoop .

### Rodar image haddop
- docker run --rm --hostname=quickstart.cloudera --privileged=true -itd -v $(pwd):/app -p 8888:8888 -p 7180:7180 -p 8181:80 -p 8088:8088 --name freep_hadoop  freep_hadoop /usr/bin/custom_quickstart.sh

- docker run --rm -m 4G --memory-reservation 2G --memory-swap 8G --hostname=quickstart.cloudera --privileged=true -itd -v $(pwd):/app --publish-all=true --name hadoop_freep  hadoop_freep /usr/bin/docker-quickstart

- /etc/init.d/ntpd start
- /home/cloudera/cloudera-manager --express

### Verificar quais portas estão sendo usadas pelo HUE e pelo YARN
- docker inspect hadoop_freep

### Comandos Haddop
- hadoop fs -mkdir -p user/freep/
- hadoop fs -mkdir -p user/freep/input
- hadoop fs -put /app/data.csv /user/freep/input

- hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
            -file /app/freep_hadoop/recommender_mapper.py -mapper /app/freep_hadoop/recommender_mapper.py \
            -file /app/freep_hadoop/recommender_reducer.py -reducer /app/freep_hadoop/recommender_reducer.py  \
            -input /user/freep/input/partitions.txt  -output /user/freep/output

- hadoop fs -rm -r /user/freep/output

-  hdfs dfsadmin -report