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

# lendo dados originais, preciso desse float precision pra ele não arredondar
data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

# print("\n\n##########ESTRATÉGIA COM TODAS AS PARTIÇÕES#########\n\n")
# start = time.time()

# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'num_aligns'
# # minhas preferências
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
#                            columns=['model1', 'prob1', 'model2', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, FullPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, FullPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)


# recommender = RankFeatureRecommender(X, y, FullPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# print("\n##################################################\n")
# # # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'model2'
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
#                            columns=['model1', 'prob1', 'num_aligns', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, FullPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, FullPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, FullPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# end = time.time()
# elapsed = end - start
# print("***********TEMPO TOTAL: ", elapsed,'**********\n')

# print("##########ESTRATÉGIA COM 50% DAS PARTIÇÕES#########\n")
# start = time.time()

# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'num_aligns'
# # minhas preferências
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
#                            columns=['model1', 'prob1', 'model2', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, PercentagePartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, PercentagePartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, PercentagePartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# print("\n##################################################\n")
# # # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'model2'
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
#                            columns=['model1', 'prob1', 'num_aligns', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, PercentagePartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, PercentagePartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, PercentagePartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# end = time.time()
# elapsed = end - start
# print("***********TEMPO TOTAL: ", elapsed,'**********\n')

# print("##########ESTRATÉGIA COM L2-NORM 50% DAS PARTIÇÕES#########\n")
# start = time.time()

# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'num_aligns'
# # minhas preferências
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
#                            columns=['model1', 'prob1', 'model2', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, L2NormPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, L2NormPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, L2NormPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# print("\n##################################################\n")
# # # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'model2'
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
#                            columns=['model1', 'prob1', 'num_aligns', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, L2NormPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, L2NormPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, L2NormPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# end = time.time()
# elapsed = end - start
# print("***********TEMPO TOTAL: ", elapsed,'**********\n')

# print("##########ESTRATÉGIA COM PCA 50% DAS PARTIÇÕES#########\n")
# start = time.time()

# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'num_aligns'
# # minhas preferências
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
#                            columns=['model1', 'prob1', 'model2', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, PCAPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# print("\n##################################################\n")
# # # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'model2'
# preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
#                            columns=['model1', 'prob1', 'num_aligns', 'prob2'])

# X = data[data.columns.intersection(preferences.columns)]
# y = data[feature]

# print("Preferências: ", preferences)
# recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=KNeighborsClassifier(n_neighbors=3))
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por KNN para", feature, 'é', recomendation)

# recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, PCAPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# end = time.time()
# elapsed = end - start
# print("***********TEMPO TOTAL: ", elapsed,'**********')


feature = 'prob1'
preferences = pd.DataFrame([['WAG+G', 10.0]],
                           columns=['model2', 'num_aligns'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = RegressorFeatureRecommender(X, y, PCAPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNR para", feature, 'é', recomendation)

recommender = RegressorFeatureRecommender(X, y, PCAPartitioner(), regressor=SVR())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por SVR para", feature, 'é', recomendation)

recommender = RegressorFeatureRecommender(X, y, PCAPartitioner(), regressor=LinearRegression())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por LinearRegression para", feature, 'é', recomendation)
# recommender = ClassifierFeatureRecommender(X, y, PCAPartitioner(), classifier=SVC())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por SVM para", feature, 'é', recomendation)

# recommender = RankFeatureRecommender(X, y, PCAPartitioner())
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# end = time.time()
# elapsed = end - start
# print("***********TEMPO TOTAL: ", elapsed,'**********')