import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from recommenders.custom_knn_feature_recommender import CustomKNNFeatureRecommender
from recommenders.rank_feature_recommender import RankFeatureRecommender
from recommenders.classifier_feature_recommender import ClassifierFeatureRecommender
from recommenders.regressor_feature_recommender import RegressorFeatureRecommender
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from partitioners.full_partitioner import FullPartitioner
from partitioners.percentage_partitioner import PercentagePartitioner
from partitioners.l2_norm_partitioner import L2NormPartitioner
from partitioners.pca_partitioner import PCAPartitioner
from preprocessors.encoding_processor import EncodingProcessor
import time
from preferences.preference import Preference
from preferences.parameter import Parameter
from preferences.value import Value

# lendo dados originais, preciso desse float precision pra ele não arredondar
data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)


# feature = 'num_aligns'

# preferences = [
#     "( model1 == 'WAG+G' ) | ( model1 == 'WAG+I+F' )",
#     "prob1 >= 1500"
#     ]
# y = data[feature]
# X = data.drop(feature, axis=1)

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, FullPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, FullPartitioner(), classifier=SVC(probability=True))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# print("\n####################################\n")

feature = 'model2'
preferences = [
    "( model1 == 'WAG+G' ) | ( model1 == 'WAG+I+F' )",
    "prob1 >= 1500",
    "length > 350"
    ]
y = data[feature]
X = data.drop(feature, axis=1)

print("Preferências: ", preferences)
recommender = RankFeatureRecommender(X, y, PCAPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=SVC(probability=True))
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, PercentagePartitioner(), classifier=SVC(probability=True))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

print("\n####################################\n")

feature = 'prob1'
preferences = [
    "( model1 == 'WAG+G' ) | ( model1 == 'WAG+I+F' )",
    "num_aligns == 10"
    ]

y = data[feature]
X = data.drop(feature, axis=1)

# print("Preferências: ", preferences)
# recommender = RegressorFeatureRecommender(X, y, PercentagePartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNR para", feature, 'é', recomendation)

