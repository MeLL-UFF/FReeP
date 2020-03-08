from ..utils.preference_processor import is_parameter_in_preferences
from ..partitioners import commons
from ..utils.preference_processor import encoded_columns_in_original_columns


def partition(X, y, columns_in_preferences):
    combinations = commons.powerset(columns_in_preferences)
    partitions = []

    for combination in combinations:
        if len(combination)>1:
            combination_encoded_columns = encoded_columns_in_original_columns(
                combination,columns_in_preferences, X.columns)
        partitions.append(X[combination_encoded_columns].columns)

    return partitions

def all_columns_present(partition, columns):
    return all(elem in partition for elem in columns)

def horizontal_filter(X, y, preferences, weights=[]):
    return commons.horizontal_filter(X, y, preferences, weights)

def vertical_filter(X, columns):
    return commons.vertical_filter(X, columns)