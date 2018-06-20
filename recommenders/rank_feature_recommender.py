import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
import numbers
from recommenders.feature_recommender import FeatureRecommender
import numpy as np


class PreferenceRelation():
    def __init__(self, classes, classifier, encoder):
        self.classes = classes
        self.classifier = classifier
        self.encoder = encoder


class RankFeatureRecommender(FeatureRecommender):

    def __init__(self, data, partitioner, weights=[], neighbors= FeatureRecommender.NEIGHBORS):
        super(RankFeatureRecommender, self).__init__(data, partitioner, weights, neighbors)

    def recommender(self, data, feature, preferences, weights):
        # as classes do meu problema são os valores da feature que quero recomendar
        classes = data[feature].unique()
        # combino dois a dois das classes
        classes_pairs = list(itertools.combinations(classes, 2))
        preferences_relations = []
        for c1, c2 in classes_pairs:
            # só me importa os dados que pertença a uma das classes
            # TODO tenho que verificar como adicionar a galera que não pertence
            preferences = pd.concat([data[data[feature] == c1], data[data[feature] == c2]])
            X = preferences.loc[:, preferences.columns != feature]
            y = preferences.loc[:, preferences.columns == feature]
            if len(X) >= self.neighbors:
                neigh = KNeighborsClassifier(n_neighbors=self.neighbors)
                X_ = X.astype(str)
                # One-hot encoding
                X_ = pd.get_dummies(X_, prefix_sep='_dummy_')
                # todas as novas colunas após o encoding
                X_encoder = list(X_)
                y = y.values.ravel()
                if isinstance(y[0], float):
                    y = [str(y_) for y_ in y]
                neigh.fit(X_.values, y)
                # guardo o modelo gerado para esse par
                preferences_relations.append(PreferenceRelation((c1, c2), neigh, X_encoder))
        # inicializo os 'votos' zerados
        voting_classes = dict.fromkeys(classes, 0)
        # recupero a probabilidade de predição de cada classificador que utilizou a classe em questão
        # e somo a sua 'votação' os 'votos' desse classificador
        for class_ in classes:
            for relation in preferences_relations:
                if class_ in relation.classes:
                    sorted_classes = sorted(relation.classes)
                    instance = self.convert_instance(relation.encoder, X.iloc[0, :])
                    prob_sorted_by_classes = relation.classifier.predict_proba([instance])[0]
                    if sorted_classes.index(class_) == 0:
                        voting_classes[class_] += prob_sorted_by_classes[0]
                    else:
                        voting_classes[class_] += prob_sorted_by_classes[1]
        #nessa partição só existe um valor possível, então é 100% de certeza
        if len(classes_pairs) <= 0:
            voting_classes[[*voting_classes.keys()][0]] = 1
        return self.rank(voting_classes)

    def recomendation(self, votes):
        # BordaCount
        # import pdb
        # pdb.set_trace()
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
                    already_voted +=[(k, v) for (k, v) in draw_votes]
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
        if self.label_encoder is None:
            return votes
        else:
            try:
                #só um no rank
                decode = self.label_encoder.inverse_transform(votes[0][0])
                return [(decode,votes[0][1])]
            except:
                ##rank com mais de um elemento
                rank_ = []
                for candidate in votes:
                    candidate_decoded = self.label_encoder.inverse_transform(candidate[0])
                    rank_.append((candidate_decoded, candidate[1]))
                self.label_encoder = None
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

# TODO problema dessa abordagem é o errar por não possuir dados da classe correta em um dos classificadores binarios
