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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from partitioners.full_partitioner import FullPartitioner
from partitioners.percentage_partitioner import PercentagePartitioner
from partitioners.l2_norm_partitioner import L2NormPartitioner
from partitioners.pca_partitioner import PCAPartitioner
from preprocessors.encoding_processor import EncodingProcessor
import time
from sklearn import decomposition
from sklearn.cluster import KMeans
import random
from functools import reduce
from get_smarties import Smarties
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score


def model_type(X, y, partitioner):
    if is_numeric_dtype(y):
        return RegressorFeatureRecommender(X, y, partitioner,
                                           regressor=SVR())
    else:
        return ClassifierFeatureRecommender(X, y, partitioner,
                                            classifier=SVC(probability=True))

def preference_to_append(y, column, value, std= False):
        if is_numeric_dtype(y):
            if std:
                std = y.std()
                return '( '+str(column) + " <= " + str(value + std) + ' ) | ' \
                    + '( '+ str(column) + " >= " + str(value - std) + ' )'
            else:
                return str(column) + " == " + str(value) 
        else:
            return str(column) + " == '" + str(value) + "'"

data = pd.read_csv('data.csv', float_precision='round_trip')
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

gs = Smarties()
dummie_data = gs.fit_transform(data)
pca = decomposition.PCA(n_components=2)
pca.fit(dummie_data)
X_transformed = pca.transform(dummie_data)

kmeans = KMeans(n_clusters=11)
kmeans.fit(X_transformed)

originals_cluster = []
samples_cluster = []
for i in range(50):
    # escolher exemplo aleatório da proveniencia
    sample = data.sample(random_state=99)
    idx = sample.index[0]
    # guardar o cluster original
    original_cluster = kmeans.predict(X_transformed[idx].reshape(1, -1))
    originals_cluster.append(original_cluster[0])
    # gerar subconjunto aleatório de parametros a recomendar
    sets = reduce(lambda result, x: result + [subset + [x] for subset in result],
                  data.columns, [[]])
    subsets = [set_ for set_ in sets if len(set_) > 1 and len(set_) < 5]
    subset = random.sample(subsets,1)[0]
    #gerar preferencias a partir das colunas restantes
    preferences_columns = list(set(data.columns) - set(subset))
    preferences = []
    for column in preferences_columns:
        preferences.append(preference_to_append(data[column],column,sample[column].values[0]))
    params_recommendations = {}
    # recomendar cada um dos parametros e guardar
    for param in subset:
        y = data[param]
        X = data.drop(list(set(data.columns)- set(preferences_columns)), axis=1)
        recommender = model_type(X, y, PCAPartitioner())
        resp = recommender.recommend(param, preferences)
        params_recommendations[param] = resp[0]
        preferences.append(preference_to_append(data[param],param,resp[0], std= True))
        preferences_columns.append(param)
    # guardar cluster que instancia gerada cai
    for param in subset:
        sample[param] = params_recommendations[param]
    dummie_sample = gs.transform(sample)
    transformed_dummie = pca.transform(dummie_sample)
    sample_cluster = kmeans.predict(transformed_dummie)
    samples_cluster.append(sample_cluster[0])
# calcular acurácia
accuracy = accuracy_score(originals_cluster, samples_cluster)
print('Acurácia', accuracy)