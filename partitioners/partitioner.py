from abc import ABC, abstractmethod
from functools import reduce
from utils.preference_processor import PreferenceProcessor


class Partitioner(ABC):

    def vertical_partition(self, X, current_preferences, preferences_parameters):
        current_preferences_parameters = PreferenceProcessor.parameters_in_preferences(
            current_preferences, X.columns.values)
        diff = list(set(X.columns.values) - set(preferences_parameters))
        current_columns = current_preferences_parameters + diff
        return X[X.columns.intersection(current_columns)]

    def horizontal_partition(self, X, y, current_preferences, weights=[]):
        X_ = X.copy()
        y_ = y.copy()
        weights_ = weights.copy()
        for preference in current_preferences:
            X_ = X_[eval(PreferenceProcessor.preference_for_eval(
                preference, X.columns.values))]
            # filtro os registros que possuem os mesmos valores das preferências
            # indices dos pesos associados a essa partição
            if len(weights) > 0:
                weights_ = weights.loc[X_.index]
            y_ = y.loc[X_.index]
        return X_, y_, weights_

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
