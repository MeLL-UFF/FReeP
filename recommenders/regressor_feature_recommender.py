# source activate py35
import pandas as pd
import numpy as np
from recommenders.feature_recommender import FeatureRecommender
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor


class RegressorFeatureRecommender(FeatureRecommender):
    def __init__(self, X, y, partitioner, weights=[],
                 regressor=KNeighborsRegressor(n_neighbors=FeatureRecommender.NEIGHBORS)):
        super(RegressorFeatureRecommender, self).__init__(
            X, y, partitioner, weights)
        self.regressor = regressor

    def recommender(self, X, y, feature, preferences, weights):
        """ X, y e preferences one-hot encoding"""
        # se só tem uma classe
        if len(np.unique(y.values)) == 1:
            return self.marjority_prediction(X, y)
        else:
            return self.regression_prediction(X,y,preferences)
        
    def recomendation(self, votes):
        l = [vote[0] for candidates in votes for vote in candidates]
        c = Counter(l)
        resp = c.most_common(1)[0][0]
        return (resp, None)

    def process_vote(self, votes):
        decode = self.preprocessor.decode_y(votes[0][0])
        return [(decode, votes[0][1])]

    def regression_prediction(self, X, y, preferences):
        self.regressor.fit(X.values, y.values)
        instance = super(RegressorFeatureRecommender,
                         self).to_predict_instance(X, preferences)
        pred = self.regressor.predict([instance])[0]
        return [(pred, None)]

    def marjority_prediction(self, X, y):
        (values,counts) = np.unique(y.values,return_counts=True)
        ind=np.argmax(counts)
        return [(values[ind], None)]
