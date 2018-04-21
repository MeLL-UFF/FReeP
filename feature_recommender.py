# source activate py35
from functools import reduce
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from numbers import Number


class FeatureRecommender(ABC):
    NEIGHBORS = 3

    def __init__(self, data):
        self.data = data

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
            #adiciono as colunas que não fazem parte nem das preferencias nem é feature
            if not_in_preferences:
                partition_set.extend(not_in_preferences)
            # adiciono a coluna que quero recomendar na ultima coluna
            partition_set.append(feature)
            # todos os registros apenas com as features deste conjunto
            partition = self.data[partition_set]
            for item in preference_set:
                # filtro os registros que possuem os mesmos valores das preferências
                partition = partition[partition[item] == preferences[item]]
            # pode ser que não tenha nenhuma proveniencia que obdeca os filtros de dados
            if len(partition) > 0:
                if not isinstance(partition[feature].values[0], Number):
                    self.label_encoder = LabelEncoder()
                    partition[feature] = self.label_encoder.fit_transform(partition[feature].values)
                else:
                    self.label_encoder = None
                vote = self.recommender(partition, feature, preferences)
                if self.label_encoder is None:
                    votes.append(vote)
                else:
                    try:
                        votes.append(self.label_encoder.inverse_transform(vote))
                        self.label_encoder = None
                    except:
                        votes.append(self.label_encoder.inverse_transform(vote[0][0]))
                        self.label_encoder = None
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


#TODO Atualizar o scikit-learn quando sair proxima release para parar de dar warning do numpy
import warnings
warnings.simplefilter("ignore", DeprecationWarning)