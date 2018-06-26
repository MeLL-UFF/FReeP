from functools import reduce
from abc import ABC, abstractmethod


class Partitioner(ABC):

    def vertical_partition(self, X, preferences_columns):
        # uma gambiarra pra passar mais de um argumento pro escopo da list comprehension
        def y(columns, preferences_columns):
            return [column for column in columns if column.split('_')[0] in preferences_columns]
        columns = y(X.columns, preferences_columns)
        # todos os registros apenas com as features deste conjunto e array de pesos (vazio inicialmente)
        return X[columns]

    def horizontal_partition(self, X, y, current_preferences, preferences, weights=[]): 
        X_ = X.copy()
        y_ = y.copy()
        weights_ = weights.copy()
        for preference in current_preferences:
            index = X[X[preference] == preferences[preference][0]].index
            # filtro os registros que possuem os mesmos valores das preferências
            # indices dos pesos associados a essa partição
            if len(weights) > 0:
                weights_ = weights.loc[index]
            X_ = X.loc[index]
            y_ = y.loc[index]
        return X_, y_, weights_

    @abstractmethod
    def partition(self, X, preferences_columns):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        """Essa funcao deve retornar as combinacoes de preferencias"""
        pass
