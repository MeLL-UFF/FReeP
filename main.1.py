import numpy as np
import pandas as pd
from recommenders.custom_knn_feature_recommender import CustomKNNFeatureRecommender
from recommenders.knn_feature_recommender import KNNFeatureRecommender
from recommenders.rank_feature_recommender import RankFeatureRecommender
from partitioners.full_partitioner import FullPartitioner
from partitioners.percentage_partitioner import PercentagePartitioner
from partitioners.l2_norm_partitioner import L2NormPartitioner
from preprocessors.encoding_processor import EncodingProcessor
import time

print("\n\n##########ESTRATÉGIA COM TODAS AS PARTIÇÕES#########\n\n")
start = time.time()

# lendo dados originais, preciso desse float precision pra ele não arredondar
data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'num_aligns'
# minhas preferências
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
                           columns=['model1', 'prob1', 'model2', 'prob2'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, FullPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(X, y, FullPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

print("\n##################################################\n")
# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'model2'
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
                           columns=['model1', 'prob1', 'num_aligns', 'prob2'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, FullPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(X, y, FullPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

end = time.time()
elapsed = end - start
print("***********TEMPO TOTAL: ", elapsed,'**********\n')

print("##########ESTRATÉGIA COM 50% DAS PARTIÇÕES#########\n")
start = time.time()

# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'num_aligns'
# minhas preferências
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
                           columns=['model1', 'prob1', 'model2', 'prob2'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, PercentagePartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(X, y, PercentagePartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

print("\n##################################################\n")
# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'model2'
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
                           columns=['model1', 'prob1', 'num_aligns', 'prob2'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, PercentagePartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(X, y, PercentagePartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

end = time.time()
elapsed = end - start
print("***********TEMPO TOTAL: ", elapsed,'**********\n')

print("##########ESTRATÉGIA COM L2-NORM 50% DAS PARTIÇÕES#########\n")
start = time.time()

# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'num_aligns'
# minhas preferências
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
                           columns=['model1', 'prob1', 'model2', 'prob2'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, L2NormPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(X, y, L2NormPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

print("\n##################################################\n")
# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'model2'
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 10.0, 1588.4588012017]],
                           columns=['model1', 'prob1', 'num_aligns', 'prob2'])

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, L2NormPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(X, y, L2NormPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

end = time.time()
elapsed = end - start
print("***********TEMPO TOTAL: ", elapsed,'**********')