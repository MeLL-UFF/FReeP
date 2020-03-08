import itertools
from tqdm import tqdm
from functools import reduce
from abc import abstractmethod
from numbers import Number
import logging
logging.basicConfig(format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing.imputation import Imputer

from ..utils.preference_processor import is_parameter_in_preferences
from ..utils.preference_processor import parameters_in_preferences
from ..utils.preference_processor import parameter_from_encoded_parameter
from ..utils.preference_processor import get_preferences_for_partition
from ..utils.encode_decode_processor import encode
from ..partitioners.commons import all_columns_present

# Feature é uma coluna de um dataFrame
# Preferences é array com pandas conditions


def recommend(X, y, feature, preferences, partitioner, model, min_neighbors,
              recommender, process_vote, recomendation):
    columns_in_preferences = parameters_in_preferences(
        preferences, X.columns.values)

    # one-hot encoding
    logging.debug('Encoding data...')
    X_encoded, y_encoded, y_encoder = encode(X, y)

    logging.debug('Generating partitions...')
    partitions_for_recommender = partitioner.partition(
        X_encoded, y_encoded, columns_in_preferences
    )
    logging.debug('Partitions generated...')
    votes = []

    logging.debug('Iterating over partitions...')
    for partition in tqdm(partitions_for_recommender):
        if len(partition) > 1:
            preferences_for_partition = get_preferences_for_partition(
                X, partition, preferences)
            # aplicar o filtro das preferencias no X e y originais
            X_, y_, weights_ = partitioner.horizontal_filter( 
                X, y, preferences_for_partition)
            # logging.debug('Data to this partition: %s', str(X))
            # tenho dados após o filtro horizontal?
            if len(X_) >= min_neighbors:
                # codificar X e y resultantes
                X_encoded, y_encoded, y_encoder = encode(X_, y_)
                # todas as colunas da partição atual estão no X_encoded?
                if all_columns_present(partition, X_encoded.columns):
                    X_partition = partitioner.vertical_filter(X_encoded, partition)
                    vote = recommender(
                        X_partition, y_encoded, feature, partition, model
                    )
                    processed_vote = process_vote(vote, y_encoder)
                    votes.append(processed_vote)
    
    logging.debug('Votes: %s', str(votes))
    if votes:
        return recomendation(votes)
    else:
        return None


def to_predict_instance(X, partition_columns):
    values_for_preferences = []
    for column in partition_columns:
        if is_parameter_in_preferences(column, partition_columns):
            values_for_preferences.append(list(X[column].unique()))
    all_combinations = list(itertools.product(*values_for_preferences))

    instances = []
    for combination in all_combinations:
        instance = []
        for column in X.columns:
            # se é um parametro dentro das preferencias
            if is_parameter_in_preferences(column, partition_columns):
                instance.append(
                    combination[list(partition_columns).index(column)])
            # se não está nas preferencias e esta codificado
            elif len(column.split("#")) > 1:
                instance.append(0)
            # se não está nas preferencias e não esta codificado
            else:
                instance.append(np.nan)
        imputer = Imputer(missing_values=np.nan, strategy="mean", axis=0)
        imputer = imputer.fit(X)
        instance = imputer.transform([instance])[0]
        instances.append(instance)
    return instances


def softmax(values):
    # exp just calculates exp for all elements in the matrix
    exp = np.exp(values)
    return exp / exp.sum(0)


def rank(votes):
        # ordeno o dicionario pelos valores, trazendo o rank
    rank = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return rank


def convert_instance(X_encoder, instance):
    test = []
    # X é codificado como One-Hot encoding, entao todas as colunas sao 0 ou 1
    for item in X_encoder:
        # O pd.get_dummies cria colunas do tipo coluna_valor
        label = item.split('_dummy_')[0]
        value = item.split('_dummy_')[1]
        # crio a instância para classificação no formato do One-Hot encoding
        if isinstance(instance[label], Number):
            if instance[label] == float(value):
                test.append(1)
            else:
                test.append(0)
        elif instance[label] == value:
            test.append(1)
        else:
            test.append(0)
    return test


# @abstractmethod
# def recommender(X, y, feature, partition_columns, model):
#     """Primitive operation. You HAVE TO override me, I'm a placeholder."""
#     pass


# @abstractmethod
# def recomendation(votes):
#     """Primitive operation. You HAVE TO override me, I'm a placeholder."""
#     pass


# @abstractmethod
# def process_vote(votes, y_encoder):
#     """Primitive operation. You HAVE TO override me, I'm a placeholder."""
#     pass
