import time
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

with open('results.txt', 'w') as f:
    for train_index, test_index in kf.split(data):
        for neighbor in neighbors:
            knn_recommender = KNNFeatureRecommender(data.iloc[train_index], neighbors=neighbor)
            rank_recommender = RankFeatureRecommender(data.iloc[train_index], neighbors=neighbor)
            knn_true_label = []
            knn_pred = []
            rank_true_label = []
            rank_pred = []
            start = time.time()
            random_feature = random.choice(list(data.columns))
            for idx in test_index:
                preferences = data.iloc[idx].to_dict()
                true_value = preferences[random_feature]
                del preferences[random_feature]
                knn_recomendation = knn_recommender.recommend(random_feature, preferences)
                rank_recomendation = rank_recommender.recommend(random_feature, preferences)
                if knn_recomendation:
                    knn_true_label.append(true_value)
                    knn_pred.append(knn_recomendation)
                if rank_recomendation:
                    rank_true_label.append(true_value)
                    rank_pred.append(rank_recomendation)
            end = time.time()
            elapsed = end - start
            print("Tempo (segundos): ", elapsed)

            f.write('FEATURE: ' + random_feature + "\n")
            f.write('%s \t %s \t %s \t %s \t %s\n' % ("MODELO", "K", "ACURACIA", "PRECISAO", "RECALL"))

            print("************KNN***********")
            knn_pred_label = [str(pred[0]) for pred in knn_pred]
            knn_true_label = [str(elem) for elem in knn_true_label]
            knn_accuracy = accuracy_score(knn_true_label, knn_pred_label)
            knn_precision = precision_score(knn_true_label, knn_pred_label, average='micro')
            knn_recall = recall_score(knn_true_label, knn_pred_label, average='micro')
            print(knn_true_label)
            print(knn_pred_label)
            print("Acurácia: ", knn_accuracy)
            print("Precisão: ", knn_precision)
            print("Recall: ", knn_recall)

            f.write('%s \t %d \t %f \t %f \t %f\n' % ('KNN', neighbor, knn_accuracy, knn_precision, knn_recall))
            f.write('TRUE_LABEL : ' + ', '.join([str(elem) for elem in knn_true_label]))
            f.write("\n")
            f.write('PRED_LABEL : ' + ', '.join([str(elem) for elem in knn_pred_label]))
            f.write("\n")

            f.write("\n")

            print("************RANK***********")
            rank_pred_label = [str(pred[0]) for pred in rank_pred]
            rank_true_label = [str(elem) for elem in rank_true_label]
            print(rank_true_label)
            print(rank_pred_label)
            rank_accuracy = accuracy_score(rank_true_label, rank_pred_label)
            rank_precision = precision_score(rank_true_label, rank_pred_label, average='micro')
            rank_recall = recall_score(rank_true_label, rank_pred_label, average='micro')
            print("Acurácia: ", accuracy_score(rank_true_label, rank_pred_label))
            print("Precisão: ", precision_score(rank_true_label, rank_pred_label, average='micro'))
            print("Recall: ", recall_score(rank_true_label, rank_pred_label, average='micro'))

            f.write('%s \t %d \t %f \t %f \t %f\n' % ('RANK', neighbor, rank_accuracy, rank_precision, rank_recall))
            f.write('TRUE_LABEL : ' + ', '.join([str(elem) for elem in rank_true_label]))
            f.write("\n")
            f.write('PRED_LABEL : ' + ', '.join([str(elem) for elem in rank_pred_label]))
            f.write("\n")
            f.write('%s %f\n' % ("TEMPO: ", elapsed))
        f.write("###################")