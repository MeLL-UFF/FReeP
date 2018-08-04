class PreferenceProcessor():

    def __init__(self):
        pass

    @staticmethod
    def preference_for_eval(preference, parameters):
        terms = preference.split(' ')
        resp = []
        for term in terms:
            if term in parameters:
                resp.append("X."+term)
            else:
                resp.append(term)
        return ''.join(resp)

    @staticmethod
    def parameters_in_preferences(preferences, parameters):
        resp = []
        for preference in preferences:
            terms = preference.split(' ')
            for term in terms:
                if term in parameters:
                    resp.append(term)
        return list(set(resp))

    @staticmethod
    def is_parameter_in_preferences(parameter, preferences):
        parameter = parameter.split('#')[0]
        return parameter in preferences

    @staticmethod
    def parameter_from_encoded_parameter(parameter):
        return parameter.split('#')[0]

    @staticmethod
    def encoded_columns_in_original_columns(current_columns_in_preferences, all_columns_in_preferences,
                                            encoded_columns):
        resp = []
        for encoded_column in encoded_columns:
            if PreferenceProcessor.is_parameter_in_preferences(encoded_column, current_columns_in_preferences):
                resp.append(encoded_column)
            elif not PreferenceProcessor.is_parameter_in_preferences(encoded_column, all_columns_in_preferences):
                resp.append(encoded_column)
        return resp
