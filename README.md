# FReep - Feature Recommender from Preferences


### Build Image

docker build -t freep .

### Run for development

docker run --rm -v ${PWD}:/app -w /app -it freep bash

### Gerar a lib
- python3 setup.py sdist bdist_wheel

### Criar imagem Hadoop

- docker build -f docker/hadoop/Dockerfile -t hadoop_freep .

### Rodar image haddop
- docker run --rm  -v ${PWD}:/app -w /app -it hadoop_freep /etc/bootstrap.sh -bash

#### Configurar path do haddop
- source .bashrc