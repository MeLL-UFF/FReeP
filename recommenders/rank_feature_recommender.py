import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
import numbers
from recommenders.feature_recommender import FeatureRecommender
import numpy as np


class PreferenceRelation():
    def __init__(self, classes, classifier):
        self.classes = classes
        self.classifier = classifier


class RankFeatureRecommender(FeatureRecommender):

    def __init__(self, X, y, partitioner, weights=[], neighbors=FeatureRecommender.NEIGHBORS):
        super(RankFeatureRecommender, self).__init__(
            X, y, partitioner, weights, neighbors)

    def recommender(self, X, y, feature, preferences, weights):
        # as classes do meu problema são os valores da feature que quero recomendar
        classes = list(set(y.values))
        # combinação dois a dois das classes
        classes_pairs = list(itertools.combinations(classes, 2))
        preferences_relations = []
        for c1, c2 in classes_pairs:
            # só me importa os dados que pertença a uma das classes
            # TODO tenho que verificar como adicionar a galera que não pertence# uma gambiarra pra passar mais de um argumento pro escopo da list comprehension
            def indexes(values, c1, c2):
                return [i for i, y in enumerate(values) if y == c1 or y == c2]
            index = indexes(y.values, c1, c2)
            X_ = X.iloc[index]
            y_ = y.iloc[index]
            if len(X_) >= self.neighbors:
                neigh = KNeighborsClassifier(n_neighbors=self.neighbors)
                neigh.fit(X_.values, y_.values)
                preferences_relations.append(
                    PreferenceRelation((c1, c2), neigh))
        # inicializo os 'votos' zerados
        voting_classes = dict.fromkeys(classes, 0)
        voting_classes_number = dict.fromkeys(classes, 0)
        # recupero a probabilidade de predição de cada classificador que utilizou a classe em questão
        # e somo a sua 'votação' os 'votos' desse classificador
        for class_ in classes:
            for relation in preferences_relations:
                if class_ in relation.classes:
                    sorted_classes = sorted(relation.classes)
                    instance = super(RankFeatureRecommender,
                                     self).to_predict_instance(X, preferences)
                    prob_sorted_by_classes = relation.classifier.predict_proba([instance])[
                        0]
                    if sorted_classes.index(class_) == 0:
                        voting_classes[class_] += prob_sorted_by_classes[0]
                    else:
                        voting_classes[class_] += prob_sorted_by_classes[1]
                    voting_classes_number[class_] += 1
        # nessa partição só existe um valor possível, então é 100% de certeza
        if len(classes_pairs) <= 0:
            voting_classes[[*voting_classes.keys()][0]] = 1
        else:
            for class_ in list(voting_classes.keys()):
                # tornando a soma de probabilidades entre 0 e 1 de novo
                voting_classes[class_] = voting_classes[class_] / \
                    voting_classes_number[class_]
            # normalizando as probabilidades
            normalized_probs = self.softmax(list(voting_classes.values()))
            for idx, class_ in enumerate(list(voting_classes.keys())):
                voting_classes[class_] = normalized_probs[idx]
        return self.rank(voting_classes)

    def recomendation(self, votes):
        #TODO corrigir esse BordaCount pra devolver probabilidades entre 0 e 1!
        classes_set = set([t[0] for rank in votes for t in rank])
        classes = dict.fromkeys(classes_set, 0)
        already_voted = []
        for rank in votes:
            weight = len(rank) - 1
            for vote in rank:
                def y(actual_vote, votes):
                    return [vote for vote in votes if vote[1] == actual_vote[1]]
                if vote not in already_voted:
                    draw_votes = y(vote, rank)
                    for candidate, percentage in draw_votes:
                        classes[candidate] += np.power(2, weight) * percentage
                    already_voted += [(k, v) for (k, v) in draw_votes]
                # import pdb
                # pdb.set_trace()
                # try:
                # except:
                #     draw_votes = [elem[0] for elem in draw_votes]
                # already_voted += [(k, v) for (k, v) in rank if (k, v) not in draw_votes[0]]
                weight -= 1
            already_voted = []
        ordered_preferences = self.rank(classes)
        resp = ordered_preferences[0][0]
        confidence = ordered_preferences[0][1] / float(len(votes))
        return (resp, confidence)

    def process_vote(self, votes):
        if len(votes) == 1:
            decode = self.preprocessor.decode_y(votes[0][0])
            return [(decode, votes[0][1])]
        else:
            rank_ = []
            for candidate in votes:
                decode = self.preprocessor.decode_y(candidate[0])
                rank_.append((decode, candidate[1]))
            return rank_

    def rank(self, votes):
        # ordeno o dicionario pelos valores, trazendo o rank
        rank = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return rank

    def convert_instance(self, X_encoder, instance):
        test = []
        # X é codificado como One-Hot encoding, entao todas as colunas sao 0 ou 1
        for item in X_encoder:
            # O pd.get_dummies cria colunas do tipo coluna_valor
            label = item.split('_dummy_')[0]
            value = item.split('_dummy_')[1]
            # crio a instância para classificação no formato do One-Hot encoding
            if isinstance(instance[label], numbers.Number):
                if instance[label] == float(value):
                    test.append(1)
                else:
                    test.append(0)
            elif instance[label] == value:
                test.append(1)
            else:
                test.append(0)
        return test

    def softmax(self, values):
        # exp just calculates exp for all elements in the matrix
        exp = np.exp(values)
        return exp / exp.sum(0)

# TODO problema dessa abordagem é o errar por não possuir dados da classe correta em um dos classificadores binarios
