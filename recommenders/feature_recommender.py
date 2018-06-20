# source activate py35
from functools import reduce
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from numbers import Number
import numpy as np


class FeatureRecommender(ABC):
    NEIGHBORS = 3

    def __init__(self, data, partitioner, weights=[], neighbors = NEIGHBORS):
        self.data = data
        self.weights = weights
        self.neighbors = neighbors
        self.partitioner = partitioner

    # Feature é uma coluna de um dataFrame
    # Preferences é um dictionary
    def recommend(self, feature, preferences):
        columns = list(preferences.keys())
        columns.append(feature)
        not_preference_set = list(set(self.data.columns) - set(columns))
        preferences_partitions = self.partitioner.partition(preferences.keys())
        votes = []
        for preference_set in preferences_partitions:
            partition, partition_weights = self.partitioner.vertical_partition(preference_set.copy(), not_preference_set, feature)
            partition, partition_weights = self.partitioner.horizontal_partition(partition, preference_set.copy(), self.weights)
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(partition) >= self.neighbors:
                partition[feature] = self.categorical_transformation(partition[feature])
                partition_weights = self.weights_selection(partition_weights)
                vote = self.recommender(partition, feature, preferences, partition_weights)
                processed_vote = self.process_vote(vote)
                votes.append(processed_vote)
        if votes:
            return self.recomendation(votes)
        else:
            return None

    @abstractmethod
    def recommender(self, data, feature, preferences, weights):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass

    @abstractmethod
    def recomendation(self, votes):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass
    
    @abstractmethod
    def process_vote(self, votes):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass

    def categorical_transformation(self, feature_columns):
        # se a feature a recomendar nao for atributo categorico transformo em atributo categorico
        if not isinstance(feature_columns.values[0], Number):
            self.label_encoder = LabelEncoder()
            feature_columns = self.label_encoder.fit_transform(feature_columns.values)
        else:
            self.label_encoder = None
        return feature_columns
    
    def weights_selection(self, partition_weights):
        if len(self.weights) > 0:
            # pesos a partir dos indices de pesos
            partition_weights = [self.weights[i] for i in partition_weights]
        return partition_weights

# TODO Atualizar o scikit-learn quando sair proxima release para parar de dar warning do numpy
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
