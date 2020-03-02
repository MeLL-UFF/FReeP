import numpy as np

from ..partitioners import commons

def partition(X, y, preferences_columns, percentile):
    combinations = commons.powerset(preferences_columns)
    cols_count = len(X[preferences_columns].columns)
    maximum_partitions = int(cols_count * (percentile / 100.0))
    partitions = {}
    data_norm = np.linalg.norm(X.values)
    for combination in combinations:
        partition = X[combination]
        partition_norm = np.linalg.norm(partition.values)
        diff_norm = data_norm - partition_norm
        if len(partitions) < maximum_partitions:
            partitions[diff_norm] = combination
        else:
            sorted_diff_norms = sorted(list(partitions.keys()), reverse=True)
            if sorted_diff_norms[0] > diff_norm:
                del partitions[sorted_diff_norms[0]]
                partitions[diff_norm] = combination
    return list(partitions.values())

def horizontal_filter(X, y, preferences, weights=[]):
    return commons.horizontal_filter(X, y, preferences, weights)

def vertical_filter(X, columns):
    return commons.vertical_filter(X, columns)