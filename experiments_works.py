import time
import json
import csv
import random
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
        return 'Percentage'
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


data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

kf = KFold(n_splits=5)

categorical_features = ['model1', 'model2']

classifiers = [KNeighborsClassifier(n_neighbors=3),
               KNeighborsClassifier(n_neighbors=5),
               KNeighborsClassifier(n_neighbors=7),
               SVC(probability=True),
               MLPClassifier(solver='sgd', hidden_layer_sizes=(6,), random_state=1)]

partitioners = [
    FullPartitioner(),
    PCAPartitioner(),
    PercentagePartitioner()
]

print("CLASSIFICAÇÃO")

with open('results/categorical_results' + time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv', 'w') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(
        ['FEATURE', 'CLASSIFIER', 'PARTITIONER', 'ACCURACY', 'TIME'])
    for feature in categorical_features:
        print("FEATURE %s" % feature)
        for classifier in classifiers:
            print("CLASSIFIER %s" % classifier)
            for partitioner in partitioners:
                start = time.time()
                for train_index, test_index in kf.split(data):
                    train_data = data.iloc[train_index]
                    y = data[feature]
                    X = data.drop(feature, axis=1)
                    true_label = []
                    pred_label = []

                    for idx in test_index:
                        preferences = data.iloc[idx].to_dict()
                        true_value = preferences[feature]
                        del preferences[feature]
                        resp = []
                        for key, value in preferences.items():
                            if type(value) is str:
                                resp.append(str(key) + " == " +
                                            "'"+str(value)+"'")
                            else:
                                resp.append(str(key) + ' == ' + str(value))
                        recommender = ClassifierFeatureRecommender(X, y, partitioner,
                                                                   classifier=classifier)
                        recomendation = recommender.recommend(
                            feature, resp)
                        if recomendation != None: 
                            true_label.append(true_value)
                            pred_label.append(recomendation[0])
                end = time.time()
                elapsed = end - start
                accuracy = accuracy_score(true_label, pred_label)
                clf_name = classifier_name(classifier)
                part_name = partitioner_name(partitioner)
                row = [feature, clf_name, part_name, accuracy, elapsed]
                print(row)
                writer.writerow(row)

print("############################")
print("REGRESSÃO")

numerical_features = ['num_aligns', 'length', 'prob1', 'prob2']
regressors = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3),
    KNeighborsRegressor(n_neighbors=5),
    KNeighborsRegressor(n_neighbors=7),
    SVR(),
    MLPRegressor(solver='sgd', hidden_layer_sizes=(6,), random_state=1)]

with open('results/numerical_results' + time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv', 'w') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(
        ['FEATURE', 'REGRESSOR', 'PARTITIONER', 'ACCURACY', 'TIME'])
    for feature in numerical_features:
        print("FEATURE %s" % feature)
        for regressor in regressors:
            for partitioner in partitioners:
                start = time.time()
                for train_index, test_index in kf.split(data):
                    train_data = data.iloc[train_index]
                    y = data[feature]
                    X = data.drop(feature, axis=1)
                    true_label = []
                    pred_label = []

                    for idx in test_index:
                        preferences = data.iloc[idx].to_dict()
                        true_value = preferences[feature]
                        del preferences[feature]
                        resp = []
                        for key, value in preferences.items():
                            if type(value) is str:
                                resp.append(str(key) + " == " +
                                            "'"+str(value)+"'")
                            else:
                                resp.append(str(key) + ' == ' + str(value))
                        recommender = RegressorFeatureRecommender(X, y, partitioner,
                                                                   regressor=regressor)
                        recomendation = recommender.recommend(
                            feature, resp)
                        true_label.append(true_value)
                        pred_label.append(recomendation[0])
                end = time.time()
                elapsed = end - start
                mse = mean_squared_error(true_label, pred_label)
                regr_name = regressor_name(regressor)
                part_name = partitioner_name(partitioner)
                row = [feature, regr_name, part_name, mse, elapsed]
                print(row)
                writer.writerow(row)
