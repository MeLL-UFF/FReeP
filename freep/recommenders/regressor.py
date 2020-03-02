from collections import Counter

import pandas as pd
import numpy as np

from ..recommenders import commons
from ..utils.encode_decode_processor import decode_y

def recommend(X, y, feature, preferences, partitioner, model,
              min_neighbors):
    return commons.recommend(X, y, feature, preferences, partitioner, model, min_neighbors,
                             recommender, process_vote, recomendation)

def recommender(X, y, feature, preferences, regressor):
    """ X, y e preferences one-hot encoding"""
    # se só tem uma classe
    if len(np.unique(y.values)) == 1:
        return marjority_prediction(X, y)
    else:
        return regression_prediction(X, y, preferences, regressor)

def recomendation(votes):
    l = [vote[0] for candidates in votes for vote in candidates]
    c = Counter(l)
    resp = c.most_common(1)[0][0]
    return (resp, None)

def process_vote(votes, y_encoder):
    decode = decode_y(votes[0][0], y_encoder)
    return [(decode, votes[0][1])]

def regression_prediction(X, y, preferences, regressor):
    regressor.fit(X.values, y.values)
    instances = commons.to_predict_instance(X, preferences)
    predictions = regressor.predict(instances)
    return [(prediction, None) for prediction in predictions]

def marjority_prediction(X, y):
    (values, counts) = np.unique(y.values, return_counts=True)
    ind = np.argmax(counts)
    return [(values[ind], None)]
