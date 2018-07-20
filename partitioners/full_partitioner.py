from partitioners.partitioner import Partitioner

class FullPartitioner(Partitioner):
    def __init__(self):
        super(FullPartitioner, self).__init__()

    def partition(self, X, y, preferences_columns, preferences_parameters):
        return super(FullPartitioner, self).powerset(preferences_columns)