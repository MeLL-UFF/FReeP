# source activate py35
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from recommenders.feature_recommender import FeatureRecommender
from collections import Counter


class KNNFeatureRecommender(FeatureRecommender):
    def __init__(self, X, y, partitioner, weights=[], neighbors=FeatureRecommender.NEIGHBORS):
        super(KNNFeatureRecommender, self).__init__(
            X, y, partitioner, weights, neighbors)

    def recommender(self, X, y, feature, preferences, weights):
        """ X, y e preferences one-hot encoding"""
        neigh = KNeighborsClassifier(n_neighbors=self.neighbors)
        neigh.fit(X.values, y.values)
        instance = super(KNNFeatureRecommender, self).to_predict_instance(X, preferences)
        pred = neigh.predict([instance])[0]
        prob = max(neigh.predict_proba([instance])[0])
        return [(pred, prob)]

    def recomendation(self, votes):
        l = [vote[0] for candidates in votes for vote in candidates]
        c = Counter(l)
        resp = c.most_common(1)[0][0]
        confidence = c.most_common(1)[0][1] / float(len(votes))
        return (resp, confidence)

    def process_vote(self, votes):
        decode = self.preprocessor.decode_y(votes[0][0])
        return [(decode, votes[0][1])]
