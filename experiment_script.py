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
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=7),
        SVC(probability=True),
        MLPClassifier(solver='sgd', hidden_layer_sizes=(6,),
                      random_state=1)
    ]


def partitioners():
    return [
        PCAPartitioner,
        PercentagePartitioner,
        # FullPartitioner()
    ]


def percentiles():
    return [30, 50, 70]


def regressors():
    return [
        LinearRegression(),
        KNeighborsRegressor(n_neighbors=3),
        KNeighborsRegressor(n_neighbors=5),
        KNeighborsRegressor(n_neighbors=7),
        SVR(),
        MLPRegressor(solver='sgd', hidden_layer_sizes=(6,), random_state=1)
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


def run_classifier(data, categorical_features, split_number, categorical_result_path):
    kf = KFold(n_splits=split_number)
    with open(categorical_result_path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(
            ['FEATURE', 'CLASSIFIER', 'PARTITIONER', 'ACCURACY', 'PRECISION', 'RECALL', 'TIME'])
        # for feature in categorical_features:
        #     for classifier in  classifiers():
        #         for partitioner in  classifiers():
        #             for percentile in  percentiles():
        paramlist = list(itertools.product(classifiers(), partitioners(),
                                           percentiles(), [kf], [data], categorical_features))

        pool = mp.Pool()
        res = pool.map(classifier_execution, paramlist)
        for row in res:
            for validation in row:
                if len(validation) > 0:
                    writer.writerow(validation)

                #  classifier_execution(classifier, partitioner, percentile, kf, data,
                #                           feature, writer)


# def classifier_execution(  classifier, partitioner, percentile, kf, data, feature, writer):


def classifier_execution(params):
    classifier = params[0]
    partitioner = params[1]
    percentile = params[2]
    kf = params[3]
    data = params[4]
    feature = params[5]
    partitioner = partitioner(percentile)

    start = time.time()
    true_label = []
    pred_label = []
    validations_results = []
    for train_index, test_index in kf.split(data):
        X, y = train_data(data, feature, train_index)
        true_validation = []
        pred_validation = []
        for idx in test_index:
            preferences = data.iloc[idx].to_dict()
            true_value = preferences[feature]
            del preferences[feature]
            resp = join_preferences_key_values(
                preferences)
            preferences_number = randint(2, len(resp))
            remove_preferences_number = len(
                resp) - preferences_number
            for i in range(remove_preferences_number):
                random_number = preferences_number = randint(
                    0, len(resp) - 1)
                del resp[random_number]
            recommender = ClassifierFeatureRecommender(X, y, partitioner,
                                                       classifier=classifier)
            recomendation = recommender.recommend(
                feature, resp)
            if recomendation != None:
                true_label.append(true_value)
                pred_label.append(recomendation[0])
                true_validation.append(true_value)
                pred_validation.append(recomendation[0])
        end_validation = time.time()
        elapsed_validation = end_validation - start
        acc_validation = accuracy_score(true_validation, pred_validation)
        precision_validation = precision_score(true_validation, pred_validation,average='weighted')
        recall_validation = recall_score(true_validation, pred_validation,average='weighted')
        clf_validation = classifier_name(classifier)
        part_validation = partitioner_name(partitioner) + '-' + str(percentile)
        validations_results.append([feature, clf_validation, part_validation, acc_validation, precision_validation, recall_validation, elapsed_validation])
    end = time.time()
    # elapsed = end - start
    # if len(true_label) > 0:
    #     accuracy = accuracy_score(true_label, pred_label)
    #     precision = precision_score(true_label, pred_label,average='weighted')
    #     recall = recall_score(true_label, pred_label,average='weighted')
    #     clf_name = classifier_name(classifier)
    #     part_name = partitioner_name(partitioner) + '-' + str(percentile)
    #     return [feature, clf_name, part_name, accuracy, precision, recall, elapsed]
    return validations_results


def run_regressors(data, numerical_features, split_number, numerical_result_path):
    kf = KFold(n_splits=split_number)
    with open(numerical_result_path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(
            ['FEATURE', 'REGRESSOR', 'PARTITIONER', 'MSE', 'TIME'])
        paramlist = list(itertools.product(regressors(), partitioners(),
                                           percentiles(), [kf], [data], numerical_features))
        pool = mp.Pool()
        res = pool.map(regressor_execution, paramlist)
        for row in res:
              for validation in row:
                if len(validation) > 0:
                    writer.writerow(validation)
                # for feature in numerical_features:
                #     for regressor in  regressors():
                #         for partitioner in  partitioners():


def regressor_execution(params):
    regressor = params[0]
    partitioner = params[1]
    percentile = params[2]
    kf = params[3]
    data = params[4]
    feature = params[5]
    partitioner = partitioner(percentile)
    start = time.time()
    true_label = []
    pred_label = []
    validations_results = []

    for train_index, test_index in kf.split(data):
        X, y = train_data(data, feature, train_index)
        true_validation = []
        pred_validation = []
        for idx in test_index:
            preferences = data.iloc[idx].to_dict()
            true_value = preferences[feature]
            del preferences[feature]
            resp = join_preferences_key_values(
                preferences)
            preferences_number = randint(2, len(resp))
            remove_preferences_number = len(
                resp) - preferences_number
            for i in range(remove_preferences_number):
                random_number = preferences_number = randint(
                    0, len(resp) - 1)
                del resp[random_number]
            recommender = RegressorFeatureRecommender(X, y, partitioner,
                                                      regressor=regressor)
            recomendation = recommender.recommend(
                feature, resp)
            if recomendation != None:
                true_label.append(true_value)
                pred_label.append(recomendation[0])
                true_validation.append(true_value)
                pred_validation.append(recomendation[0])
        end_validation = time.time()
        elapsed_validation = end_validation - start
        mse_validation = mean_squared_error(true_label, pred_label)
        regr_validation = regressor_name(regressor)
        part_validation = partitioner_name(partitioner) + '-' + str(percentile)
        validations_results.append([feature, regr_validation, part_validation, mse_validation, elapsed_validation])

    # if len(true_label) > 0:
    #     mse = mean_squared_error(true_label, pred_label)
    #     regr_name = regressor_name(regressor)
    #     part_name = partitioner_name(partitioner) + '-' + str(percentile)
    #     return [feature, regr_name, part_name, mse, elapsed]
    return validations_results
