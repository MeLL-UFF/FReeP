import time
import json
import csv
import random
import pandas as pd
from custom_knn_feature_recommender import CustomKNNFeatureRecommender
from knn_feature_recommender import KNNFeatureRecommender
from rank_feature_recommender import RankFeatureRecommender
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import itertools
import importlib
import multiprocessing as mp


def neighbors():
    return [3, 5, 7]

def execution(params):
    neighbor = params[0]
    kf = params[1]
    data = params[2]
    feature = params[3]
    _class = params[4]
    submodule = ""
    if _class == "KNNFeatureRecommender":
        submodule = "knn_feature_recommender"
    else:
        submodule = "rank_feature_recommender"

    start = time.time()
    true_label = []
    pred_label = []
    # Load "module.submodule.MyClass"
    MyClass = getattr(importlib.import_module(submodule), _class)
    for train_index, test_index in kf.split(data):
        recommender = MyClass(
            data.iloc[train_index], neighbors=neighbor)
        for idx in test_index:
            preferences = data.iloc[idx].to_dict()
            true_value = preferences[feature]
            del preferences[feature]
            recomendation = recommender.recommend(
                feature, preferences)
            if recomendation:
                true_label.append(true_value)
                pred_label.append(recomendation)
    end = time.time()
    elapsed = end - start
    pred_label = [str(pred[0]) for pred in pred_label]
    true_label = [str(elem) for elem in true_label]
    precision = precision_score(
        true_label, pred_label, average='weighted')
    recall = recall_score(
        true_label, pred_label, average='weighted')
    return [feature, _class, precision, recall, elapsed]

def run(data, path):
    # data = pd.read_csv('data.csv', float_precision='round_trip')
    # # apenas os dados de sucesso, sem a coluna de erro
    # data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

    features = list(data.columns)
    kf = KFold(n_splits=5)
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(
            ['FEATURE', 'MODEL', 'PRECISION', 'RECALL', 'TIME'])
        paramlist = list(itertools.product(neighbors(), [kf], [data],
                                           features, ["KNNFeatureRecommender", "RankFeatureRecommender"]))
        pool = mp.Pool(1)
        res = pool.map(execution, paramlist)
        for row in res:
            writer.writerow(row)

sciphy = pd.read_csv('sciphy.csv', float_precision='round_trip')
sciphy = sciphy[~sciphy['erro']].copy().drop('erro', axis=1).reset_index(drop=True)
path = 'results/sciphy/original'+ \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'
run(sciphy, path)

montage = pd.read_csv('montage.csv', float_precision='round_trip')
columns = ['cntr', 'ra', 'dec', 'cra', 'cdec', 'crval1', 'crval2', 'crota2']
montage = montage[columns]
path = 'results/montage/original'+ \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'
run(montage, path)