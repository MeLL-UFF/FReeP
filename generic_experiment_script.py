import time
import json
import csv
import random
from random import randint
import multiprocessing as mp
import itertools
import pandas as pd
from recommenders.classifier_feature_recommender import ClassifierFeatureRecommender
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from recommenders.regressor_feature_recommender import RegressorFeatureRecommender
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from partitioners.full_partitioner import FullPartitioner
from partitioners.percentage_partitioner import PercentagePartitioner
from partitioners.pca_partitioner import PCAPartitioner
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from recommenders.multi_recommendation import MultiRecommendation
from pandas.api.types import is_numeric_dtype
from statistics import mean


def partitioner_name(partitioner):
    if type(partitioner) is PCAPartitioner:
        return 'PCA'
    elif type(partitioner) is PercentagePartitioner:
        return 'ANOVA'
    else:
        return 'Full'


def classifier_name(classifier):
    if type(classifier) is SVC:
        return 'SVM'
    elif type(classifier) is MLPClassifier:
        return 'Multi Layer Perceptron'
    else:
        # KNN
        if classifier.n_neighbors == 3:
            return 'KNN 3'
        elif classifier.n_neighbors == 5:
            return 'KNN 5'
        else:
            return 'KNN 7'


def regressor_name(regressor):
    if type(regressor) is SVR:
        return 'SVR'
    elif type(regressor) is MLPRegressor:
        return 'Multi Layer Perceptron'
    elif type(regressor) is LinearRegression:
        return 'Linear Regression'
    else:
        # KNR
        if regressor.n_neighbors == 3:
            return 'KNR 3'
        elif regressor.n_neighbors == 5:
            return 'KNR 5'
        else:
            return 'KNR 7'


def classifiers():
    return [
        # KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=7),
        SVC(probability=True),
        # MLPClassifier(solver='sgd', hidden_layer_sizes=(6,),
        #   random_state=1)
    ]


def partitioners():
    return [
        # PCAPartitioner,
        PercentagePartitioner,
        # FullPartitioner()
    ]


def percentiles():
    # return [30, 50, 70]
    return [50]


def regressors():
    return [
        # LinearRegression(),
        # KNeighborsRegressor(n_neighbors=3),
        KNeighborsRegressor(n_neighbors=5),
        KNeighborsRegressor(n_neighbors=7),
        SVR(),
        # MLPRegressor(solver='sgd', hidden_layer_sizes=(6,), random_state=1)
    ]


def join_preferences_key_values(preferences):
    resp = []
    for key, value in preferences.items():
        if type(value) is str:
            resp.append("{} == '{}'".format(str(key), str(value)))
        else:
            resp.append(str(key) + ' == ' + str(value))
    return resp


def train_data(data, feature, index):
    train_data = data.iloc[index]
    y = train_data[feature]
    X = train_data.drop(feature, axis=1)
    return X, y


def run(data, result_path):
    with open(result_path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(
            ['CLASSIFIER', 'REGRESSOR', 'PARTITIONER', 'MSE', 'PRECISION', 'RECALL', 'MISSING'])
        paramlist = list(itertools.product(classifiers(), regressors(), partitioners(),
                                           percentiles(), [data]))
        pool = mp.Pool()
        results = [pool.apply_async(run_generic_recommendation, args=(params,)) for params in paramlist]
        output = [p.get() for p in results]
        for row in output:
            writer.writerow(row)
        # res = pool.map(run_generic_recommendation, paramlist)
        # for row in res:
        #     writer.writerow(row)


def run_generic_recommendation(params):
    classifier = params[0]
    regressor = params[1]
    partitioner = params[2]
    percentile = params[3]
    data = params[4]
    partitioner = partitioner(percentile)

    sample = data.sample(10)
    combined = data.append(sample)
    train = combined[~combined.index.duplicated(keep=False)]

    mses = []
    precisions = []
    recalls = []
    missing_rec = 0
    for index, record in sample.iterrows():
        m = randint(2, len(record) - 2)
        l = list(range(len(record) - 2))
        selected_idx = []
        preferences = []
        for i in range(m):
            idx = random.choice(l)
            selected_idx.append(idx)
            param = record.keys()[idx]
            val = record.values[idx]
            if is_numeric_dtype(train[param]):
                preferences.append(str(param) + ' == ' + str(val))
            else:
                preferences.append(str(param) + " == '" +
                                   str(val) + "'")

            l.remove(idx)
        remain_idx = list(set(list(range(len(record)))) - set(selected_idx))
        true_values = dict(
            list(zip(record.keys()[remain_idx], record.values[remain_idx])))
        recommender = MultiRecommendation(
            train, partitioner, classifier, regressor)
        rec = recommender.recommend(preferences)
        if(len(rec) == 0):
            missing_rec += 1
        print("Preferências: " + str(preferences))
        print("Recomendação: " + str(rec))
        num_pred = [v for k, v in rec.items() if not isinstance(v, str)]
        cat_pred = [v for k, v in rec.items() if isinstance(v, str)]
        num_true = [true_values[k]
                    for k, v in rec.items() if not isinstance(v, str)]
        cat_true = [true_values[k]
                    for k, v in rec.items() if isinstance(v, str)]
        if len(num_pred) > 0:
            mse = mean_squared_error(num_true, num_pred)
            mses.append(mse)
        if len(cat_pred) > 0:
            precision = precision_score(cat_true, cat_pred, average='weighted')
            precisions.append(precision)
            recall = recall_score(cat_true, cat_pred, average='weighted')
            recalls.append(recall)
            # accuracy = accuracy_score(cat_true, cat_pred)
            # acc.append(accuracy)
    part_name = partitioner_name(partitioner) + '-' + str(percentile)
    regr_name = regressor_name(regressor)
    class_name = classifier_name(classifier)
    if len(precisions) > 0:
        recall = mean(recalls)
        precision = mean(precisions)
        mean_error = mean(mses)
        return [class_name, regr_name, part_name, mean_error, precision, recall, missing_rec]
    else:
        return [class_name, regr_name, part_name, -1, 0, 0, missing_rec]
