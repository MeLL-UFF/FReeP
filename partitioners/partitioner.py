from functools import reduce
from abc import ABC, abstractmethod

class Partitioner(ABC):
    def __init__(self,data, preferences):
        self.data = data
        self.preferences = preferences
    
    def vertical_partition(self, preference_set, not_preference_set, feature):
        vertical_partition = preference_set
        # adiciono as colunas que não fazem parte nem das preferencias nem é feature
        if not_preference_set:
            vertical_partition.extend(not_preference_set)
        # adiciono a coluna que quero recomendar na ultima coluna
        vertical_partition.append(feature)
        # todos os registros apenas com as features deste conjunto
        # array de pesos (vazio inicialmente)
        return self.data[vertical_partition], []

    def horizontal_partition(self, partition, preference_set, weights):
        partition_weights = []
        for preference in preference_set:
                # filtro os registros que possuem os mesmos valores das preferências
                if len(weights) > 0:
                    # indices dos pesos associados a essa partição
                    partition_weights = partition[partition[preference] == self.preferences[preference]].index
                partition = partition[partition[preference] == self.preferences[preference]]
        return partition, partition_weights

    @abstractmethod
    def partition(self, lst):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        """Essa funcao deve retornar as combinacoes de preferencias"""
        pass