import numpy as np

from ..partitioners import commons
from ..utils.preference_processor import encoded_columns_in_original_columns


def partition(X, y, columns_in_preferences, percentile=70):
    combinations = commons.powerset(columns_in_preferences)

    cols_count = len(set(columns_in_preferences).intersection(
        set([x.split("#")[0] for x in X.columns.tolist()])))

    maximum_partitions = int(cols_count * (percentile / 100.0))
    partitions = {}
    data_norm = np.linalg.norm(X.values)
    for combination in combinations:

        if len(combination) > 1:
            combination_encoded_columns = encoded_columns_in_original_columns(
                combination,columns_in_preferences, X.columns)

            X_partition = X[combination_encoded_columns]
            partition_norm = np.linalg.norm(X_partition.values)
            diff_norm = data_norm - partition_norm
            if len(partitions) < maximum_partitions:
                partitions[diff_norm] = combination_encoded_columns
            else:
                sorted_diff_norms = sorted(
                    list(partitions.keys()), reverse=True)
                if sorted_diff_norms[0] > diff_norm:
                    del partitions[sorted_diff_norms[0]]
                    partitions[diff_norm] = combination_encoded_columns
    return list(partitions.values())


def horizontal_filter(X, y, preferences, weights=[]):
    return commons.horizontal_filter(X, y, preferences, weights)


def vertical_filter(X, columns):
    return commons.vertical_filter(X, columns)
