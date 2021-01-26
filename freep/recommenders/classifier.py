from collections import Counter
import logging
import pandas as pd
import numpy as np

from ..recommenders import commons
from ..utils.encode_decode_processor import decode_y

logging.basicConfig(
    format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

def recommend(X, y, feature, preferences, partitioner, model,
              min_neighbors):
    return commons.recommend(X, y, feature, preferences, partitioner, model, min_neighbors,
                             recommender, process_vote, recomendation)


def recommender(X, y, feature, partition_columns, classifier):
    """ X, y e preferences one-hot encoding"""
    # se só tem uma classe
    if len(np.unique(y.values)) == 1:
        return marjority_prediction(X, y)
    else:
        return classifier_prediction(X, y, partition_columns, classifier)


def recomendation(votes):
    l = [vote[0] for candidates in votes for vote in candidates]
    c = Counter(l)
    resp = c.most_common(1)[0][0]
    confidence = c.most_common(1)[0][1] / float(len(votes))
    return (resp, confidence)


def process_vote(votes, y_encoder):
    decode = decode_y(votes[0][0], y_encoder)
    return [(decode, votes[0][1])]


def classifier_prediction(X, y, partition_columns, classifier):
    logging.debug('Classifier')
    logging.debug('Fitting')
    classifier.fit(X.values, y.values)
    logging.debug('Creating instances')
    instances = commons.to_predict_instance(X, partition_columns)
    logging.debug('Predicting')
    predictions = classifier.predict(instances)
    logging.debug('Predicting probabilities')
    probs = np.array([max(l) for l in classifier.predict_proba(instances)])
    logging.debug('Finish')
    return list(zip(predictions, probs))


def marjority_prediction(X, y):    
    counts = np.bincount(y.values.astype(int))
    pred = counts.argmax()
    prob = counts[pred] / len(y)
    return [(pred, prob)]
