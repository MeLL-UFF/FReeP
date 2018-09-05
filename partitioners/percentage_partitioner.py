from partitioners.partitioner import Partitioner
from sklearn.feature_selection import SelectPercentile
from preprocessors.encoding_processor import EncodingProcessor
from utils.preference_processor import PreferenceProcessor
import itertools


class PercentagePartitioner(Partitioner):
    def __init__(self, percentile=50):
        self.percentile = percentile
        super(PercentagePartitioner, self).__init__()

    def partition(self, X, y, columns_in_preferences):
        feature_selection = SelectPercentile(percentile=self.percentile)
        feature_selection.fit(X, y)
        encoded_columns = X.columns[feature_selection.get_support()].values
        parameters_in_preferences = []
        for column in encoded_columns:
            parameters_in_preferences.append(
                PreferenceProcessor.parameter_from_encoded_parameter(column))
        resp = []
        for preference in columns_in_preferences:
            # todos os parametros nesta preferencia
            parameters = PreferenceProcessor.parameters_in_preferences(
                [preference], parameters_in_preferences)
            # parametros dessa preferencia estao nas preferencias das particoes
            if all(elem in parameters_in_preferences for elem in parameters):
                resp.append(preference)
        return super(PercentagePartitioner, self).powerset(resp)

    def all_columns_present(self, partition, columns):
        return all(elem in columns for elem in partition)
