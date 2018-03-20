# source activate py35
import pandas as pd
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier


# função que cria os conjunto das partes de um array
def list_powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])


# lendo dados originais
data = pd.read_csv('data.csv')
# apenas os dados de sucesso, sem a coluna de erro
data = data.loc[data['erro'] == False].drop('erro', axis=1)
# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
proveniencia = data.drop('num_aligns', axis=1)
a_recomendar = 'num_aligns'
# dados de entrada fakes
dados_entrada = {
    'length': 237.0,
    'model1': 'WAG+G',
    'prob1': 1588.45880,
    'model2': 'WAG+G',
    'prob2': 1588.45880
}

# to removendo o conjunto vazio do conjunto das partes
conjunto_das_partes = list_powerset(proveniencia.columns)[1:]

for conjunto in conjunto_das_partes:
    conj = conjunto.copy()
    # adiciono a coluna que quero recomendar na ultima coluna
    conj.append(a_recomendar)
    prov = data[conj]
    prov_real = prov
    for item in conjunto:
        # registro apenas com as características dos dados de entrada
        prov_real = prov_real[prov_real[item] == dados_entrada[item]]
        #forço todas as colunas serem string e depois faço o one-hot encoding
        dt = prov_real.astype(str)
        #TODO preciso tirar a coluna alvo daqui! vai dar merda na hora de usar no treinamento
        dt = pd.get_dummies(dt)
        neigh = KNeighborsClassifier(n_neighbors=3)
        #X = todas as colunas menos a última, Y= última
        neigh.fit(dt.iloc[:, :-1].values, dt.iloc[:, -1].values)
        #TODO utilizar os dados corretos na predicao
        print(neigh.predict([[1]]))
    print(prov_real)
