import itertools
import copy

import pandas as pd
import numpy as np

from ..recommenders import commons
from ..utils.encode_decode_processor import decode_y


class PreferenceRelation():
    def __init__(self, classes, model):
        self.classes = classes
        self.model = model


def recommender(X, y, feature, preferences, model, min_neighbors=3):
    # as classes do meu problema são os valores da feature que quero recomendar
    classes = list(set(y.values))
    # combinação dois a dois das classes
    classes_pairs = list(itertools.combinations(classes, 2))
    preferences_relations = []
    for c1, c2 in classes_pairs:
        # só me importa os dados que pertença a uma das classes
        # TODO tenho que verificar como adicionar a galera que não pertence
        # uma gambiarra pra passar mais de um argumento pro escopo da list comprehension
        def indexes(values, c1, c2):
            return [i for i, y in enumerate(values) if y == c1 or y == c2]
        index = indexes(y.values, c1, c2)
        X_ = X.iloc[index]
        y_ = y.iloc[index]
        if len(X_) >= min_neighbors:
            model = copy.deepcopy(model)
            model.fit(X_.values, y_.values)
            preferences_relations.append(
                PreferenceRelation((c1, c2), model))
    # inicializo os 'votos' zerados
    voting_classes = dict.fromkeys(classes, 0)
    voting_classes_number = dict.fromkeys(classes, 0)
    # recupero a probabilidade de predição de cada classificador que utilizou a classe em questão
    # e somo a sua 'votação' os 'votos' desse classificador
    for class_ in classes:
        for relation in preferences_relations:
            if class_ in relation.classes:
                sorted_classes = sorted(relation.classes)
                instances = commons.to_predict_instance(X, preferences)
                probs_sorted_by_classes = relation.model.predict_proba(instances)
                if sorted_classes.index(class_) == 0:
                    voting_classes[class_] += np.sum(probs_sorted_by_classes, axis=0)[0]
                else:
                    voting_classes[class_] += np.sum(probs_sorted_by_classes, axis=0)[1]
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
        normalized_probs = commons.softmax(list(voting_classes.values()))
        for idx, class_ in enumerate(list(voting_classes.keys())):
            voting_classes[class_] = normalized_probs[idx]
    return commons.rank(voting_classes)

def recomendation(votes):
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
            weight -= 1
        already_voted = []
    ordered_preferences = rank(classes)
    resp = [recomendation[0] for recomendation in ordered_preferences]
    return resp

def process_vote(votes, y_encoder):
    if len(votes) == 1:
        decode = decode_y(votes[0][0], y_encoder)
        return [(decode, votes[0][1])]
    else:
        rank_ = []
        for candidate in votes:
            decode = decode_y(candidate[0], y_encoder)
            rank_.append((decode, candidate[1]))
        return rank_
