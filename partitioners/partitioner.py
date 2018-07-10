from functools import reduce
from abc import ABC, abstractmethod
from functools import reduce

class Partitioner(ABC):

    def vertical_partition(self, X, preferences_columns):
        return X[preferences_columns]

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
    def partition(self, X, y, preferences_columns):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        """Essa funcao deve retornar as combinacoes de preferencias"""
        pass

    def powerset(self, columns):
        """ Conjunto das partes de todas as colunas das preferências, exceto o vazio"""
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                      columns, [[]])[1:]