import time
import json
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

data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

kf = KFold(n_splits=5)

neighbors = [3, 5, 7]
features = list(data.columns)

with open('results.txt', 'w') as f:
    for feature in features:

        f.write('FEATURE = %s\n\n' % feature)
        print("FEATURE %s" % feature)

        for neighbor in neighbors:

            f.write('K = %d\n' % neighbor)
            print('K = %d' % neighbor)

            for train_index, test_index in kf.split(data):
                knn_recommender = KNNFeatureRecommender(data.iloc[train_index], neighbors=neighbor)
                rank_recommender = RankFeatureRecommender(data.iloc[train_index], neighbors=neighbor)
                knn_true_label = []
                knn_pred = []
                rank_true_label = []
                rank_pred = []
                start = time.time()

                for idx in test_index:
                    #TODO ver como posso guardar uma lista de dicts contendo as preferencias
                    preferences = data.iloc[idx].to_dict()
                    true_value = preferences[feature]
                    del preferences[feature]

                    f.write("PREFERÊNCIAS: ")
                    f.write(json.dumps(preferences))
                    f.write("\n")
                    
                    knn_recomendation = knn_recommender.recommend(feature, preferences)
                    rank_recomendation = rank_recommender.recommend(feature, preferences)
                    if knn_recomendation:
                        knn_true_label.append(true_value)
                        knn_pred.append(knn_recomendation)
                    if rank_recomendation:
                        rank_true_label.append(true_value)
                        rank_pred.append(rank_recomendation)
                end = time.time()
                elapsed = end - start

                f.write('%s \t %s \t %s \t %s\n' % ("MODELO", "ACURACIA", "PRECISAO", "RECALL"))
                print('%s \t %s \t %s \t %s' % ("MODELO", "ACURACIA", "PRECISAO", "RECALL"))

                knn_pred_label = [str(pred[0]) for pred in knn_pred]
                knn_true_label = [str(elem) for elem in knn_true_label]
                knn_accuracy = accuracy_score(knn_true_label, knn_pred_label)
                knn_precision = precision_score(knn_true_label, knn_pred_label, average='micro')
                knn_recall = recall_score(knn_true_label, knn_pred_label, average='micro')

                f.write('%s  \t %f \t %f \t %f\n' % ('KNN', knn_accuracy, knn_precision, knn_recall))
                f.write('TRUE_LABEL : ' + ', '.join([str(elem) for elem in knn_true_label]))
                f.write("\n")
                f.write('PRED_LABEL : ' + ', '.join([str(elem) for elem in knn_pred_label]))
                f.write("\n\n")

                print('%s  \t %f \t %f \t %f' % ('KNN', knn_accuracy, knn_precision, knn_recall))
                print('TRUE_LABEL : ' + ', '.join([str(elem) for elem in knn_true_label]))
                print('PRED_LABEL : ' + ', '.join([str(elem) for elem in knn_pred_label]))
                print("\n")

                rank_pred_label = [str(pred[0]) for pred in rank_pred]
                rank_true_label = [str(elem) for elem in rank_true_label]
                rank_accuracy = accuracy_score(rank_true_label, rank_pred_label)
                rank_precision = precision_score(rank_true_label, rank_pred_label, average='micro')
                rank_recall = recall_score(rank_true_label, rank_pred_label, average='micro')

                f.write('%s \t %d \t %f \t %f \t %f\n' % ('RANK', neighbor, rank_accuracy, rank_precision, rank_recall))
                f.write('TRUE_LABEL : ' + ', '.join([str(elem) for elem in rank_true_label]))
                f.write("\n")
                f.write('PRED_LABEL : ' + ', '.join([str(elem) for elem in rank_pred_label]))
                f.write("\n\n")
                f.write('%s %f\n' % ("TEMPO: ", elapsed))
                f.write("\n\n")

                print('%s \t %d \t %f \t %f \t %f\n' % ('RANK', neighbor, rank_accuracy, rank_precision, rank_recall))
                print('TRUE_LABEL : ' + ', '.join([str(elem) for elem in rank_true_label]))
                print("\n")
                print('PRED_LABEL : ' + ', '.join([str(elem) for elem in rank_pred_label]))
                print("\n\n")
                print('%s %f\n' % ("TEMPO: ", elapsed))
                print("\n\n")
            f.write("####################################\n")
            print("######################################")