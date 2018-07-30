# source activate py35
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from recommenders.feature_recommender import FeatureRecommender
from collections import Counter


class ClassifierFeatureRecommender(FeatureRecommender):
    def __init__(self, X, y, partitioner, weights=[],
                 classifier=KNeighborsClassifier(n_neighbors=FeatureRecommender.NEIGHBORS)):
        super(ClassifierFeatureRecommender, self).__init__(
            X, y, partitioner, weights)
        self.classifier = classifier

    def recommender(self, X, y, feature, preferences, weights):
        """ X, y e preferences one-hot encoding"""
        # se só tem uma classe
        if len(np.unique(y.values)) == 1:
            return self.marjority_prediction(X, y)
        else:
            return self.classifier_prediction(X,y,preferences)
        
    def recomendation(self, votes):
        l = [vote[0] for candidates in votes for vote in candidates]
        c = Counter(l)
        resp = c.most_common(1)[0][0]
        confidence = c.most_common(1)[0][1] / float(len(votes))
        return (resp, confidence)

    def process_vote(self, votes):
        decode = self.preprocessor.decode_y(votes[0][0])
        return [(decode, votes[0][1])]

    def classifier_prediction(self, X, y, preferences):
        self.classifier.fit(X.values, y.values)
        instances = super(ClassifierFeatureRecommender,
                         self).to_predict_instance(X, preferences)
        predictions = self.classifier.predict(instances)
        probs = np.array([max(l) for l in self.classifier.predict_proba(instances)])
        return list(zip(predictions, probs))

    def marjority_prediction(self, X, y):
        counts = np.bincount(y.values.astype(int))
        pred = counts.argmax()
        prob = counts[pred] / len(y)
        return [(pred, prob)]
