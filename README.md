# FReep - Feature Recommender from Preferences


### Build Image

docker build -f docker/original/Dockerfile -t freep .

### Run for development

docker run --rm -v ${PWD}:/app -w /app -it freep bash

### Gerar a lib
- python3 setup.py sdist bdist_wheel

### Criar imagem Hadoop

- docker build -f docker/hadoop/Dockerfile -t hadoop_freep .

### Rodar image haddop
- docker run --rm -m 4G --memory-reservation 2G --memory-swap 8G --hostname=quickstart.cloudera --privileged=true -t -i -v $(pwd):/app --publish-all=true -p8888 -p8088 --name hadoop_freep  hadoop_freep /usr/bin/docker-quickstart

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

- hadoop fs -put -f /app/mapred-site.xml /user/root/input/mapred-site.xml