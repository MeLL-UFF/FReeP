import random
import pandas as pd
from custom_knn_feature_recommender import CustomKNNFeatureRecommender
from knn_feature_recommender import KNNFeatureRecommender
from rank_feature_recommender import RankFeatureRecommender
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

kf = KFold(n_splits=2)

for train_index, test_index in kf.split(data):
    recommender = KNNFeatureRecommender(data.iloc[train_index])
    knn_true = []
    knn_pred = []
    for idx in test_index:
        preferences = data.iloc[idx].to_dict()
        random_feature = random.choice(list(preferences))
        true_value = preferences[random_feature]
        knn_true.append(true_value)
        del preferences[random_feature]
        recomendation = recommender.recommend(random_feature, preferences)
        knn_pred.append(recomendation)
    print(knn_true)
    print(knn_pred)