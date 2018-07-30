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
        parameter = parameter.split('_')[0]
        return parameter in preferences    
    
    @staticmethod
    def parameter_from_encoded_parameter(parameter):
        return parameter.split('_')[0]