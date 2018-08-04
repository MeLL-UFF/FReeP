# source activate py35
from functools import reduce
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from numbers import Number
import numpy as np
from preprocessors.encoding_processor import EncodingProcessor
from utils.preference_processor import PreferenceProcessor
import itertools
from sklearn.preprocessing.imputation import Imputer


class FeatureRecommender(ABC):
    NEIGHBORS = 3

    def __init__(self, X, y, partitioner, weights=[], neighbors=NEIGHBORS):
        self.X = X
        self.y = y
        self.weights = weights
        self.neighbors = neighbors
        self.partitioner = partitioner

    # Feature é uma coluna de um dataFrame
    # Preferences é array com pandas conditions
    def recommend(self, feature, preferences):
        X_, y_, weights_ = self.partitioner.horizontal_filter(
            self.X, self.y, preferences)
        if len(X) > FeatureRecommender.NEIGHBORS:
            self.columns_in_preferences = PreferenceProcessor.parameters_in_preferences(
                preferences, self.X.columns.values)
            self.preprocessor = EncodingProcessor()
            # one-hot encoding
            X_encoded, y_encoded = self.preprocessor.encode(
                X_, y_)
            partitions_for_recommender = self.partitioner.partition(
                X_encoded, y_encoded, self.columns_in_preferences)
            votes = []
            for partition in partitions_for_recommender:
                X_partition = self.partitioner.vertical_filter(
                    X_encoded,  partition)
                if len(X_partition) >= self.neighbors:
                    vote = self.recommender(
                        X_partition, y_encoded, feature, partition, weights_)
                    processed_vote = self.process_vote(vote)
                    votes.append(processed_vote)
            if votes:
                return self.recomendation(votes)
            else:
                return None
        return None

    def to_predict_instance(self, X, partition_columns):
        values_for_preferences = []
        for column in partition_columns:
            if PreferenceProcessor.is_parameter_in_preferences(column, partition_columns):
                values_for_preferences.append(list(X[column].unique()))
        all_combinations = list(itertools.product(
            *values_for_preferences))

        instances = []
        for combination in all_combinations:
            instance = []
            for column in X.columns:
                # se é um parametro dentro das preferencias
                if PreferenceProcessor.is_parameter_in_preferences(column, partition_columns):
                    instance.append(
                        combination[list(partition_columns).index(column)])
                # se não está nas preferencias e esta codificado
                elif len(column.split("#")) > 1:
                    instance.append(0)
                # se não está nas preferencias e não esta codificado
                else:
                    instance.append(np.nan)
            imputer = Imputer(
                missing_values=np.nan, strategy='mean', axis=0)
            imputer = imputer.fit(X)
            instance = imputer.transform([instance])[0]
            instances.append(instance)
        return instances

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


# TODO Atualizar o scikit-learn quando sair proxima release para parar de dar warning do numpy
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
