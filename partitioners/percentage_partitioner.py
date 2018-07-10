from partitioners.partitioner import Partitioner
from sklearn.feature_selection import SelectPercentile
from functools import reduce


class PercentagePartitioner(Partitioner):
    def __init__(self, percentile=50):
        self.percentile = percentile
        super(PercentagePartitioner, self).__init__()

    def partition(self, X, y, preferences_columns):
        feature_selection = SelectPercentile(percentile=self.percentile)
        feature_selection.fit(X, y)
        columns = X.columns[feature_selection.get_support()].values
        return super(PercentagePartitioner, self).powerset(preferences_columns)
