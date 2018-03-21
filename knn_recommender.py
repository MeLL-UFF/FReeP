# source activate py35
import pandas as pd
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# função que cria os conjunto das partes de um array
def list_powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])


# lendo dados originais, preciso desse float precision pra ele não arredondar
dados = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
dados = dados.loc[dados['erro'] == False].drop('erro', axis=1)
# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
a_recomendar = 'num_aligns'
features = dados.drop(a_recomendar, axis=1)
# dados de entrada fakes
dados_entrada = {
    'length': 237.0,
    'model1': 'WAG+G',
    'prob1': 1588.4588012017,
    'model2': 'WAG+G',
    'prob2': 1588.4588012017
}

# to removendo o conjunto vazio do conjunto das partes
conjunto_das_partes = list_powerset(features.columns)[1:]
print("Tamanho do conjunto das partes: ", len(conjunto_das_partes))

recomendacoes = []

for parte in conjunto_das_partes:
    conjunto = parte.copy()
    # adiciono a coluna que quero recomendar na ultima coluna
    conjunto.append(a_recomendar)
    # todos os registros apenas com as features deste conjunto
    proveniencia = dados[conjunto]
    for feature in parte:
        # filtro os registros que possuem os mesmos valores dos dados de entrada
        proveniencia = proveniencia[proveniencia[feature] == dados_entrada[feature]]
    # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
    if len(proveniencia) > 0:
        # X = todas as colunas menos a última, Y= última
        X = proveniencia.iloc[:, :-1]
        y = proveniencia.iloc[:, -1]
        # forço todas as colunas serem string
        X = X.astype(str)
        # One-hot encoding
        X = pd.get_dummies(X)
        X_encoder = list(X)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X.values, y.values)
        test = []
        #X é codificado como One-Hot encoding, entao todas as colunas sao 0 ou 1
        for item in X_encoder:
            # O pd.get_dummies cria colunas do tipo coluna_valor
            rotulo = item.split('_')[0]
            valor = item.split('_')[1]
            if dados_entrada[rotulo] == valor or dados_entrada[rotulo] == float(valor):
                test.append(1)
            else:
                test.append(0)
        #guardo as recomendacoes para o meu item feito por cada classificador
        recomendacoes.append(neigh.predict([test])[0])
#o valor que for mais vezes recomendado é o vencedor
recomendacao = np.bincount(recomendacoes).argmax()
print(a_recomendar, ' = ', recomendacao)