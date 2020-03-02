from ..utils.preference_processor import is_parameter_in_preferences
from ..partitioners import commons


def partition(X, y, columns_in_preferences):
    columns = []
    for column in X.columns:
        if is_parameter_in_preferences(column, columns_in_preferences):
            columns.append(column)
    return commons.powerset(columns)

def all_columns_present(partition, columns):
    return all(elem in partition for elem in columns)

def horizontal_filter(X, y, preferences, weights=[]):
    return commons.horizontal_filter(X, y, preferences, weights)

def vertical_filter(X, columns):
    return commons.vertical_filter(X, columns)