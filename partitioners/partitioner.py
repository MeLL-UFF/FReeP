from abc import ABC, abstractmethod
from functools import reduce
from utils.preference_processor import PreferenceProcessor


class Partitioner(ABC):

    def horizontal_filter(self, X, y, preferences, weights=[]):
        X_ = X.copy()
        y_ = y.copy()
        weights_ = weights.copy()
        for preference in preferences:
            #TODO preciso saber com filtrar por float de uma forma eficiente!!!
            X_ = X_[eval(PreferenceProcessor.preference_for_eval(
                preference, X.columns.values))]
            # filtro os registros que possuem os mesmos valores das preferências
            # indices dos pesos associados a essa partição
            if len(weights) > 0:
                weights_ = weights.loc[X_.index]
            y_ = y.loc[X_.index]
        return X_, y_, weights_
    
    def vertical_filter(self, X, columns):
        return X[X.columns.intersection(columns)]

    @abstractmethod
    def partition(self, X, y, preferences_columns, preferences_parameters):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        """Essa funcao deve retornar as combinacoes de preferencias"""
        pass

    def powerset(self, columns):
        """ Conjunto das partes de todas as colunas das preferências, exceto o vazio"""
        sets = reduce(lambda result, x: result + [subset + [x] for subset in result],
                      columns, [[]])
        # apenas os conjuntos com pelo menos dois elementos
        return [set_ for set_ in sets if len(set_) > 1]
