from sklearn.decomposition import PCA
import numpy as np

from ..utils.preference_processor import encoded_columns_in_original_columns
from ..partitioners import commons

def partition(X, y, columns_in_preferences, percentile=70):
    # não quero usar todas as partições possíveis... por isso uso o PCA pra verificar quais 
    # partições mantém mais a semelhança com os dados originais
    columns_in_preferences_powerset = commons.powerset(columns_in_preferences)

    pca_original = PCA(n_components=2)
    pca_original.fit(X)
    original_similarity_metric = pca_original.singular_values_

    partitions = []
    for combination in columns_in_preferences_powerset:
        if len(combination) > 1:
            combination_encoded_columns = encoded_columns_in_original_columns(
                combination,columns_in_preferences, X.columns)
            X_partition = X[combination_encoded_columns]

            pca_partition = PCA(n_components=2)
            pca_partition.fit(X_partition)
            partition_similarity_metric = pca_partition.singular_values_

            diff_similarity_metric = np.linalg.norm(
                original_similarity_metric - partition_similarity_metric)

            partitions.append([diff_similarity_metric, X_partition.columns])

    partitions = sorted(partitions, key=lambda x: x[0])

    maximum_partitions = int(len(X.columns) * (percentile / 100.0))

    selected_partitions = partitions[:maximum_partitions]
    partitions_for_recomender = [partition[1] for partition in selected_partitions]

    return partitions_for_recomender


def all_columns_present(partition, columns):
    return all(elem in partition for elem in columns)

def horizontal_filter(X, y, preferences, weights=[]):
    return commons.horizontal_filter(X, y, preferences, weights)

def vertical_filter(X, columns):
    return commons.vertical_filter(X, columns)
