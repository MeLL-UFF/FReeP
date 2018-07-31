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
from partitioners.full_partitioner import FullPartitioner
from partitioners.percentage_partitioner import PercentagePartitioner
from partitioners.pca_partitioner import PCAPartitioner
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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


data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

kf = KFold(n_splits=5)

categorical_features = ['model1', 'model2']
numerical_features = ['num_aligns', 'length', 'prob1', 'prob2']

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

with open('results/categorical_results' + time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv', 'w') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(
                    ['FEATURE', 'CLASSIFIER', 'PARTITIONER', 'ACCURACY', 'TIME'])
    for feature in categorical_features:
        print("FEATURE %s" % feature)
        for classifier in classifiers:
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
                        true_label.append(true_value)
                        pred_label.append(recomendation[0])
                end = time.time()
                elapsed = end - start
                accuracy = accuracy_score(true_label, pred_label)
                clf_name = classifier_name(classifier)
                part_name = partitioner_name(partitioner)
                writer.writerow(
                    [feature, clf_name, part_name, accuracy, elapsed])