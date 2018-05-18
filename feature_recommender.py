# source activate py35
from functools import reduce
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from numbers import Number
import numpy as np


class FeatureRecommender(ABC):
    NEIGHBORS = 3

    def __init__(self, data, weights=[], neighbors = NEIGHBORS):
        self.data = data
        self.weights = weights
        self.neighbors = neighbors

    # Feature é uma coluna de um dataFrame
    # Preferences é um dictionary
    def recommend(self, feature, preferences):
        columns = list(preferences.keys())
        columns.append(feature)
        all_columns = list(self.data.columns)
        not_in_preferences = list(set(all_columns) - set(columns))
        # conjunto das partes de todas as colunas das preferências, exceto o vazio
        preferences_power_set = self.list_powerset(preferences.keys())[1:]
        votes = []
        for preference_set in preferences_power_set:
            partition_set = preference_set.copy()
            # adiciono as colunas que não fazem parte nem das preferencias nem é feature
            if not_in_preferences:
                partition_set.extend(not_in_preferences)
            # adiciono a coluna que quero recomendar na ultima coluna
            partition_set.append(feature)
            # todos os registros apenas com as features deste conjunto
            partition = self.data[partition_set]
            partition_weights = []
            for item in preference_set:
                # filtro os registros que possuem os mesmos valores das preferências
                if len(self.weights) > 0:
                    # indices dos pesos associados a essa partição
                    partition_weights = partition[partition[item] == preferences[item]].index
                partition = partition[partition[item] == preferences[item]]
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(partition) >= FeatureRecommender.NEIGHBORS:
                if not isinstance(partition[feature].values[0], Number):
                    self.label_encoder = LabelEncoder()
                    partition[feature] = self.label_encoder.fit_transform(partition[feature].values)
                else:
                    self.label_encoder = None
                if len(self.weights) > 0:
                    # pesos a partir dos indices de pesos
                    partition_weights = [self.weights[i] for i in partition_weights]
                vote = self.recommender(partition, feature, preferences, partition_weights)
                if self.label_encoder is None:
                    votes.append(vote)
                else:
                    try:
                        #só um no rank
                        decode = self.label_encoder.inverse_transform(vote[0][0])
                        votes.append([(decode,vote[0][1])])
                    except:
                        ##rank com mais de um elemento
                        rank_ = []
                        for candidate in vote:
                            candidate_decoded = self.label_encoder.inverse_transform(candidate[0])
                            rank_.append((candidate_decoded, candidate[1]))
                        votes.append(rank_)
                        self.label_encoder = None
        if votes:
            return self.recomendation(votes)
        else:
            return None

    # função que cria os conjunto das partes de um array
    def list_powerset(self, lst):
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                      lst, [[]])

    @abstractmethod
    def recommender(self, data, feature, preferences, weights):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass

    @abstractmethod
    def recomendation(self, votes):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass


# TODO Atualizar o scikit-learn quando sair proxima release para parar de dar warning do numpy
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
