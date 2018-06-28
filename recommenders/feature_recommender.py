# source activate py35
from functools import reduce
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from numbers import Number
import numpy as np
from preprocessors.encoding_processor import EncodingProcessor


class FeatureRecommender(ABC):
    NEIGHBORS = 3

    def __init__(self, X, y, partitioner, weights=[], neighbors=NEIGHBORS):
        self.X = X
        self.y = y
        self.weights = weights
        self.neighbors = neighbors
        self.partitioner = partitioner

    # Feature é uma coluna de um dataFrame
    # Preferences é um dictionary
    def recommend(self, feature, preferences):
        self.preprocessor = EncodingProcessor()
        X_, y_, encoded_preferences = self.preprocessor.encode(
            self.X, self.y, preferences)
        preferences_partitions = self.partitioner.partition(
            X_, preferences.keys())
        votes = []
        for current_preferences in preferences_partitions:
            X_partition = self.partitioner.vertical_partition(
                X_, current_preferences)
            y_partition = y_.copy()
            encoded_current_preferences = self.preprocessor.encode_preference(preferences, encoded_preferences,
                                                                              current_preferences)
            # current_preferences é um array
            # encoded_preferences é um dataframe
            # preferences é um dataframe
            X_partition, y_partition, weights_ = self.partitioner.horizontal_partition(X_partition, y_partition, encoded_current_preferences,
                                                                                       encoded_preferences, self.weights)

            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(X_partition) >= self.neighbors:
                vote = self.recommender(
                    X_partition, y_partition, feature, encoded_preferences, weights_)
                processed_vote = self.process_vote(vote)
                votes.append(processed_vote)
        if votes:
            return self.recomendation(votes)
        else:
            return None

    # @abstractmethod
    # def recommender(self, data, feature, preferences, weights):
    #     """Primitive operation. You HAVE TO override me, I'm a placeholder."""
    #     pass

    @abstractmethod
    def recommender(self, X, y, feature, preferences, weights):
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
            feature_columns = self.label_encoder.fit_transform(
                feature_columns.values)
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
