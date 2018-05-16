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

kf = KFold(n_splits=2)

start = time.time()
for train_index, test_index in kf.split(data):
    knn_recommender = KNNFeatureRecommender(data.iloc[train_index])
    rank_recommender = RankFeatureRecommender(data.iloc[train_index])
    knn_true_label = []
    knn_pred = []
    rank_true_label = []
    rank_pred = []
    for idx in test_index:
        preferences = data.iloc[idx].to_dict()
        random_feature = random.choice(list(preferences))
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
    knn_pred_label = [pred[0] for pred in knn_pred]
    knn_true_label = [str(elem) for elem in knn_true_label]
    # print(knn_true)
    # print(knn_pred_label)
    print("************KNN***********")
    print("Acurácia: ", accuracy_score(knn_true_label, knn_pred_label))
    print("Precisão: ", precision_score(knn_true_label, knn_pred_label, average='micro'))
    print("Recall: ", recall_score(knn_true_label, knn_pred_label, average='micro'))

    rank_pred_label = [pred[0] for pred in rank_pred]
    rank_true_label = [str(elem) for elem in rank_true_label]
    # print(knn_true)
    # print(knn_pred_label)
    print("************RANK***********")
    print("Acurácia: ", accuracy_score(rank_true_label, rank_pred_label))
    print("Precisão: ", precision_score(rank_true_label, rank_pred_label, average='micro'))
    print("Recall: ", recall_score(rank_true_label, rank_pred_label, average='micro'))

end = time.time()
elapsed = end - start
print("Tempo (segundos): ", elapsed)