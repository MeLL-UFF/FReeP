import freep
from sklearn.svm import SVR
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import random
import csv
import json
import time
import logging
logging.basicConfig(
    format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)


# data = pd.read_csv('data.csv', float_precision='round_trip')
data = pd.read_csv('data.csv', float_precision='round_trip')
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

features = ['model1', 'prob1']

classifier = SVC(probability=True)
preferences = data.sample(1).to_dict()
true_values = [preferences[feature] for feature in features]
for feature in features:
    del preferences[feature]

model_preferences = []
for key, d in preferences.items():
    for i, value in d.items():
        if type(value) is str:
            model_preferences.append("{} == '{}'".format(
                str(key), str(value)))
        else:
            model_preferences.append(str(key) + ' == ' + str(value))

y = data[feature]
X = data.drop(feature, axis=1)

logging.debug('Target: ' + feature)
logging.debug('Preferences: ' + str(model_preferences))

# resp = freep.recommenders.rank.recommend(X, y, feature, model_preferences,
#                                                freep.partitioners.percentage,
#                                                SVC(probability=True),
#                                                3)

resp = freep.recommenders.general.recommend(pd.concat([X, y], axis=1), model_preferences,
                                            freep.partitioners.pca)

print("Recomendation: " + str(resp))
