from pandas.api.types import is_numeric_dtype

def preference_for_eval(preference, parameters):
    terms = preference.split(" ")
    resp = []
    for term in terms:
        if term in parameters:
            resp.append("X." + term)
        else:
            resp.append(term)
    return "".join(resp)


def parameters_in_preferences(preferences, parameters):
    resp = []
    for preference in preferences:
        terms = preference.split(" ")
        for term in terms:
            if term in parameters:
                resp.append(term)
    return list(set(resp))


def is_parameter_in_preferences(parameter, preferences):
    parameter = parameter.split("#")[0]
    return parameter in preferences


def parameter_from_encoded_parameter(parameter):
    return parameter.split("#")[0]


def encoded_columns_in_original_columns(
    current_columns_in_preferences, 
    all_columns_in_preferences, 
    encoded_columns):
    resp = []
    for encoded_column in encoded_columns:
        if is_parameter_in_preferences(encoded_column, current_columns_in_preferences):
            resp.append(encoded_column)
        elif not is_parameter_in_preferences(
            encoded_column, all_columns_in_preferences
        ):
            resp.append(encoded_column)
    return resp

def preference_to_append(y, column, vote_value):
    if is_numeric_dtype(y):
        std = y.std()
        return '( '+str(column) + " <= " + str(vote_value + std) + ' ) | ' \
                + '( '+ str(column) + " >= " + str(vote_value - std) + ' )'

    else:
        return str(column) + " == '" + str(vote_value) + "'"

def get_preferences_for_partition(X, partition, preferences):
    preferences_for_partition = []
    for preference in preferences:
        current_preference_parameters = parameters_in_preferences(
            [preference], X.columns.values
        )

        decodeds_parameters = set(
            [parameter_from_encoded_parameter(elem) for elem in partition]
        )

        result = all(
            elem in decodeds_parameters for elem in current_preference_parameters
        )

        if result:
            preferences_for_partition.append(preference)
    return preferences_for_partition