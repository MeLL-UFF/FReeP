from partitioners.partitioner import Partitioner
from sklearn.feature_selection import SelectPercentile
from preprocessors.encoding_processor import EncodingProcessor
from utils.preference_processor import PreferenceProcessor
import itertools


class PercentagePartitioner(Partitioner):
    def __init__(self, percentile=50):
        self.percentile = percentile
        super(PercentagePartitioner, self).__init__()

    def partition(self, X, y, preferences_columns, preferences_parameters):
        feature_selection = SelectPercentile(percentile=self.percentile)
        self.preprocessor = EncodingProcessor()
        X_, y_ = self.preprocessor.encode(
            X[preferences_parameters], y)
        feature_selection.fit(X_, y)
        encoded_columns = X_.columns[feature_selection.get_support()].values
        parameters_in_preferences = []
        for column in encoded_columns:
            parameters_in_preferences.append(
                PreferenceProcessor.parameter_from_encoded_parameter(column))
        resp = []
        for preference in preferences_columns:
            #todos os parametros nesta preferencia
            parameters = PreferenceProcessor.parameters_in_preferences(
                [preference], parameters_in_preferences)
            #parametros dessa preferencia estao nas preferencias das particoes
            if all(elem in parameters_in_preferences for elem in parameters):
                resp.append(preference)
        return super(PercentagePartitioner, self).powerset(resp)
