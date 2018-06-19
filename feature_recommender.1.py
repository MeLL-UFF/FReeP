# source activate py35
from functools import reduce
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from numbers import Number
import numpy as np
from partitioners.full_partitioner import FullPartitioner


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
        columns = list(preferences.keys()).append(feature)
        not_preference_set = list(set(self.data.columns) - set(columns))
        preferences_partitions = self.partitioner.partition(preferences.keys())
        votes = []
        for preference_set in preferences_partitions:
            partition, partition_weights = self.partitioner.vertical_partition(preference_set.copy(), not_preference_set, feature)
            partition, partition_weights = self.partitioner.horizontal_partition(partition, preference_set.copy(), self.weights)
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(partition) >= self.neighbors:

                #TODO refatorar esta parte do código pra remover esses ifs malucos!
                # se a feature a recomendar nao for atributo categorico transformo em atributo categorico
                if not isinstance(partition[feature].values[0], Number):
                    self.label_encoder = LabelEncoder()
                    partition[feature] = self.label_encoder.fit_transform(partition[feature].values)
                else:
                    self.label_encoder = None
                if len(self.weights) > 0:
                    # pesos a partir dos indices de pesos
                    partition_weights = [self.weights[i] for i in partition_weights]
                vote = self.recommender(partition, feature, preferences, partition_weights)
                
                #se nao houve transformacao da feature a recomendar apenas guardo o voto
                if self.label_encoder is None:
                    votes.append(vote)
                else:
                    # caso contrario
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
