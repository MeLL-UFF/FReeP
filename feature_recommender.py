# source activate py35
import pandas as pd
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class FeatureRecommender():

    def __init__(self,data, weights=False):
        self.data = data
        self.weights = weights

    #Feature é uma coluna de um dataFrame
    #Preferences é um dictionary
    def recommend(self, feature, preferences):
        #conjunto das partes de todas as colunas das preferências, exceto o vazio
        preferences_power_set  = self.list_powerset(preferences.keys())[1:]
        votes = []
        for set in preferences_power_set:
            partition_set = set.copy()
            # adiciono a coluna que quero recomendar na ultima coluna
            partition_set.append(feature)
            # todos os registros apenas com as features deste conjunto
            partition = self.data[partition_set]
            for item in set:
                # filtro os registros que possuem os mesmos valores das preferências
                partition = partition[partition[item] == preferences[item]]
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(partition) > 0:
                # X = todas as colunas menos a última, Y= última
                X = partition.iloc[:, :-1]
                y = partition.iloc[:, -1]
                # forço todas as colunas serem string
                X = X.astype(str)
                # One-hot encoding
                X = pd.get_dummies(X)
                #todas as novas colunas após o encoding
                X_encoder = list(X)
                neigh = KNeighborsClassifier(n_neighbors=3)
                neigh.fit(X.values, y.values)
                test = []
                # X é codificado como One-Hot encoding, entao todas as colunas sao 0 ou 1
                for item in X_encoder:
                    # O pd.get_dummies cria colunas do tipo coluna_valor
                    label = item.split('_')[0]
                    value = item.split('_')[1]
                    #crio a instância para classificação no formato do One-Hot encoding
                    if preferences[label] == value or preferences[label] == float(value):
                        test.append(1)
                    else:
                        test.append(0)
                # guardo votos feitos por cada classificador
                votes.append(neigh.predict([test])[0])
        # o valor que for mais votado é o recomendado
        return np.bincount(votes).argmax()

    # função que cria os conjunto das partes de um array
    def list_powerset(self, lst):
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                      lst, [[]])
