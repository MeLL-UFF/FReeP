from partitioners.partitioner import Partitioner
from utils.preference_processor import PreferenceProcessor

class FullPartitioner(Partitioner):
    def __init__(self):
        super(FullPartitioner, self).__init__()

    def partition(self, X, y, columns_in_preferences):
        columns = []
        for column in X.columns:
            if PreferenceProcessor.is_parameter_in_preferences(column, columns_in_preferences):
                columns.append(column)
        return super(FullPartitioner, self).powerset(columns)

    def all_columns_present(self, partition, columns):
        return all(elem in partition for elem in columns)