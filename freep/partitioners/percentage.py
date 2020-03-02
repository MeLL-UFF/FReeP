import itertools
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression

from ..utils.preference_processor import parameter_from_encoded_parameter
from ..utils.preference_processor import parameters_in_preferences as check_parameters
from ..partitioners import commons


def partition(X, y, columns_in_preferences, percentile, score=f_classif):
    feature_selection = SelectPercentile(score, percentile=percentile)
    feature_selection.fit(X, y)
    encoded_columns = X.columns[feature_selection.get_support()].values
    parameters_in_preferences = []
    for column in encoded_columns:
        parameters_in_preferences.append(
            parameter_from_encoded_parameter(column))
    resp = []
    for preference in columns_in_preferences:
        # todos os parametros nesta preferencia
        parameters = check_parameters(
            [preference], parameters_in_preferences)
        # parametros dessa preferencia estao nas preferencias das particoes
        if all(elem in parameters_in_preferences for elem in parameters):
            resp.append(preference)
    return commons.powerset(resp)

def horizontal_filter(X, y, preferences, weights=[]):
    return commons.horizontal_filter(X, y, preferences, weights)

def vertical_filter(X, columns):
    return commons.vertical_filter(X, columns)