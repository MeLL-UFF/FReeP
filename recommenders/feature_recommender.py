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
    # Preferences é um dictionary
    def recommend(self, feature, preferences):
        preferences_parameters = PreferenceProcessor.parameters_in_preferences(
            preferences, self.X.columns.values)
        preferences_partitions = self.partitioner.partition(
            self.X, self.y, preferences)
        votes = []
        for current_preferences in preferences_partitions:
            # vertical partition
            X_partition = self.partitioner.vertical_partition(
                self.X, current_preferences, preferences_parameters)
            # horizontal parition
            X_partition, y_partition, weights_ = self.partitioner.horizontal_partition(
                X_partition, self.y, current_preferences, self.weights)
            # one-hot encoding
            self.filter_X = X_partition.copy()
            self.filter_y = y_partition.copy()
            self.preprocessor = EncodingProcessor()
            X_partition, y_partition = self.preprocessor.encode(
                X_partition, y_partition)
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(X_partition) >= self.neighbors:
                vote = self.recommender(
                    X_partition, y_partition, feature, current_preferences, weights_)
                processed_vote = self.process_vote(vote)
                votes.append(processed_vote)
        if votes:
            return self.recomendation(votes)
        else:
            return None

    def to_predict_instance(self, X, preferences):
        preferences_parameters = PreferenceProcessor.parameters_in_preferences(
            preferences, self.X.columns.values)
        preferences_parameters_values = []
        for parameter in preferences_parameters:
            # todos os valores possiveis para esse parametro depois da filtragem
            preferences_parameters_values.append(
                list(self.filter_X[parameter].unique()))
        all_combinations = list(itertools.product(
            *preferences_parameters_values))
        instances = []
        # X é codificado como One-Hot encoding, entao todas as colunas sao numericas
        for combination in all_combinations:
            instance = []
            for param in X:
                # se esse parametro não sofreu encoding
                if param in preferences_parameters:
                    instance.append(
                        combination[preferences_parameters.index(param)])
                # se sofreu encoding
                elif PreferenceProcessor.is_parameter_in_preferences(param, preferences_parameters):
                    decoded_param = PreferenceProcessor.parameter_from_encoded_parameter(
                        param)
                    value = combination[preferences_parameters.index(
                        decoded_param)]
                    # one-hot encoding entao tudo é 0 ou 1
                    if str(decoded_param) + "_" + str(value) == param:
                        instance.append(1)
                    else:
                        instance.append(0)
                # se é um parâmetro fora das preferências
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
