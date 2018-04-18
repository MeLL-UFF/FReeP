import pandas as pd
from knn_feature_recommender import KNNFeatureRecommender
from rank_feature_recommender import RankFeatureRecommender

# lendo dados originais, preciso desse float precision pra ele não arredondar
data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data.loc[data['erro'] == False].drop('erro', axis=1)
# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'num_aligns'
# minhas preferências
preferences = {
    'length': 237.0,
    'model1': 'WAG+G',
    'prob1': 1588.4588012017,
    'model2': 'WAG+G',
    'prob2': 1588.4588012017
}

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(data)
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(data)
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)

print("\n##################################################\n")
# to fazendo de conta que vou querer recomendar a coluna 'num_aligns'
feature = 'model2'
# minhas preferências
preferences = {
    'length': 237.0,
    'model1': 'WAG+G',
    'prob1': 1588.4588012017,
    'num_aligns': 10.0,
    'prob2': 1588.4588012017
}

print("Preferências: ", preferences)
recommender = KNNFeatureRecommender(data)
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por KNN para", feature, 'é', recomendation)

recommender = RankFeatureRecommender(data)
recomendation = recommender.recommend(feature, preferences)
print("Recomendação por RANK para", feature, 'é', recomendation)
