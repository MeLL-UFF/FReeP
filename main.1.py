import numpy as np
import pandas as pd
from recommenders.custom_knn_feature_recommender import CustomKNNFeatureRecommender
from recommenders.knn_feature_recommender import KNNFeatureRecommender
from recommenders.rank_feature_recommender import RankFeatureRecommender
from partitioners.full_partitioner import FullPartitioner
from partitioners.percentage_partitioner import PercentagePartitioner
from preprocessors.encoding_processor import EncodingProcessor

# lendo dados originais, preciso desse float precision pra ele não arredondar
data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)
# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'num_aligns'
# minhas preferências
preferences = pd.DataFrame([['WAG+G', 1588.4588012017, 'WAG+G', 1588.4588012017]],
    columns=['model1', 'prob1', 'model2', 'prob2'])

# {
#     # 'length': 237.0,
#     'model1': 'WAG+G',
#     'prob1': 1588.4588012017,
#     'model2': 'WAG+G',
#     'prob2': 1588.4588012017
# }

X = data[data.columns.intersection(preferences.columns)]
y = data[feature]

# preprocessor = EncodingProcessor()
# X_, y_, encoded_preferences = preprocessor.encode(X, y, preferences)


# partitioner = FullPartitioner()
# X_, weights = partitioner.vertical_partition(X_, list(preferences.keys()))
# X_, y_ = partitioner.horizontal_partition(X_, y_, weights, list(encoded_preferences.columns),
#  encoded_preferences)

# partitioner = PercentagePartitioner(data,feature, preferences)

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(X, y, FullPartitioner())
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

# # print("\n")
# #
# # fake_weights = np.random.randint(1,10,data.shape[0])
# # recommender = CustomKNNFeatureRecommender(data, fake_weights)
# # recomendation = recommender.recommend(feature, preferences)
# # print("Recomendação por Custom-KNN para", feature, 'é', recomendation)
# #
# # print("\n")

# recommender = RankFeatureRecommender(data, partitioner)
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)

# print("\n##################################################\n")
# # to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
# feature = 'model2'
# # minhas preferências
# preferences = {
#     # 'length': 237.0,
#     'model1': 'WAG+G',
#     'prob1': 1588.4588012017,
#     'num_aligns': 10.0,
#     'prob2': 1588.4588012017
# }

# partitioner = FullPartitioner(data,feature, preferences)

# # print("Preferências: ", preferences)
# # recommender = KNNFeatureRecommender(data)
# # recomendation = recommender.recommend(feature, preferences)
# # print("Recomendação por KNN para", feature, 'é', recomendation)
# #
# # print("\n")
# #
# # fake_weights = np.random.randint(1,10,data.shape[0])
# # recommender = CustomKNNFeatureRecommender(data, fake_weights)
# # recomendation = recommender.recommend(feature, preferences)
# # print("Recomendação por Custom-KNN para", feature, 'é', recomendation)
# #
# # print("\n")

# #
# recommender = RankFeatureRecommender(data, partitioner)
# recomendation = recommender.recommend(feature, preferences)
# print("Recomendação por RANK para", feature, 'é', recomendation)
