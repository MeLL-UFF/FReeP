# source activate py35
from functools import reduce
from abc import ABC, abstractmethod


class FeatureRecommender(ABC):

    def __init__(self, data):
        self.data = data

    # Feature é uma coluna de um dataFrame
    # Preferences é um dictionary
    def recommend(self, feature, preferences):
        # conjunto das partes de todas as colunas das preferências, exceto o vazio
        preferences_power_set = self.list_powerset(preferences.keys())[1:]
        votes = []
        for set in preferences_power_set:
            partition_set = set.copy()
            # adiciono a coluna que quero recomendar na ultima coluna
            partition_set.append(feature)
            # todos os registros apenas com as features deste conjunto
            partition = self.data[partition_set]
            for item in set:
                # filtro os registros que possuem os mesmos valores das preferências
                partition = partition[partition[item] == preferences[item]]
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(partition) > 0:
                vote = self.recommender(partition, feature, preferences)
                votes.append(vote)

        return self.recomendation(votes)

    # função que cria os conjunto das partes de um array
    def list_powerset(self, lst):
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                      lst, [[]])
    @abstractmethod
    def recommender(self, data, feature, preferences):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass

    @abstractmethod
    def recomendation(self, votes):
        """Primitive operation. You HAVE TO override me, I'm a placeholder."""
        pass