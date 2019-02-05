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


class ExperimentScript():

    def partitioner_name(self, partitioner):
        if type(partitioner) is PCAPartitioner:
            return 'PCA'
        elif type(partitioner) is PercentagePartitioner:
            return 'Percentage'
        else:
            return 'Full'

    def classifier_name(self, classifier):
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

    def regressor_name(self, regressor):
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

    def classifiers(self):
        return [
            KNeighborsClassifier(n_neighbors=3),
            KNeighborsClassifier(n_neighbors=5),
            KNeighborsClassifier(n_neighbors=7),
            SVC(probability=True),
            MLPClassifier(solver='sgd', hidden_layer_sizes=(6,),
                          random_state=1)
        ]

    def partitioners(self):
        return [
            PCAPartitioner(),
            PercentagePartitioner(),
            FullPartitioner()
        ]

    def regressors(self):
        return [
            LinearRegression(),
            KNeighborsRegressor(n_neighbors=3),
            KNeighborsRegressor(n_neighbors=5),
            KNeighborsRegressor(n_neighbors=7),
            SVR(),
            MLPRegressor(solver='sgd', hidden_layer_sizes=(6,), random_state=1)
        ]

    def join_preferences_key_values(self, preferences):
        resp = []
        for key, value in preferences.items():
            if type(value) is str:
                resp.append("{} == '{}'".format(str(key), str(value)))
            else:
                resp.append(str(key) + ' == ' + str(value))
        return resp
    
    def train_data(self, data, feature, index):
        train_data = data.iloc[index]
        y = train_data[feature]
        X = train_data.drop(feature, axis=1)
        return X, y

    def run_classifier(self, data, categorical_features, split_number, categorical_result_path):
        kf = KFold(n_splits=split_number)
        with open(categorical_result_path, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(
                ['FEATURE', 'CLASSIFIER', 'PARTITIONER', 'ACCURACY', 'TIME'])
            for feature in categorical_features:
                for classifier in self.classifiers():
                    for partitioner in self.partitioners():
                        start = time.time()
                        true_label = []
                        pred_label = []

                        for train_index, test_index in kf.split(data):
                            X, y = self.train_data(data, feature, train_index)
                            for idx in test_index:
                                preferences = data.iloc[idx].to_dict()
                                true_value = preferences[feature]
                                del preferences[feature]
                                resp = self.join_preferences_key_values(preferences)
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
                        clf_name = self.classifier_name(classifier)
                        part_name = self.partitioner_name(partitioner)
                        row = [feature, clf_name, part_name, accuracy, elapsed]
                        writer.writerow(row)

    def run_regressors(self, data, numerical_features, split_number, numerical_result_path):
        kf = KFold(n_splits=split_number)
        with open(numerical_result_path, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(
                ['FEATURE', 'REGRESSOR', 'PARTITIONER', 'MSE', 'TIME'])
            for feature in numerical_features:
                for regressor in self.regressors():
                    for partitioner in self.partitioners():
                        start = time.time()
                        for train_index, test_index in kf.split(data):
                            X, y = self.train_data(data, feature, train_index)
                            true_label = []
                            pred_label = []

                            for idx in test_index:
                                preferences = data.iloc[idx].to_dict()
                                true_value = preferences[feature]
                                del preferences[feature]
                                resp = self.join_preferences_key_values(preferences)
                                recommender = RegressorFeatureRecommender(X, y, partitioner,
                                                                          regressor=regressor)
                                recomendation = recommender.recommend(
                                    feature, resp)
                                if recomendation != None:
                                    true_label.append(true_value)
                                    pred_label.append(recomendation[0])
                        end = time.time()
                        elapsed = end - start
                        if len(true_label) > 0:
                            mse = mean_squared_error(true_label, pred_label)
                            regr_name = self.regressor_name(regressor)
                            part_name = self.partitioner_name(partitioner)
                            row = [feature, regr_name, part_name, mse, elapsed]
                            writer.writerow(row)
