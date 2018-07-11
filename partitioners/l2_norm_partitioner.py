from partitioners.partitioner import Partitioner
from sklearn.feature_selection import SelectPercentile
import numpy as np


class L2NormPartitioner(Partitioner):
    def __init__(self, percentile=50):
        self.percentile = percentile
        super(L2NormPartitioner, self).__init__()

    def partition(self, X, y, preferences_columns):
        combinations = super(L2NormPartitioner, self).powerset(
            preferences_columns)
        maximum_partitions = int(len(X[preferences_columns].columns) * (self.percentile / 100.0))
        partitions = {}
        data_norm = np.linalg.norm(X.values)
        for combination in combinations:
            partition = X[combination]
            partition_norm = np.linalg.norm(partition.values)
            diff_norm = data_norm - partition_norm
            if len(partitions) < maximum_partitions:
                partitions[diff_norm] = combination
            else:
                sorted_diff_norms = sorted(list(partitions.keys()), reverse=True)
                if sorted_diff_norms[0] > diff_norm:
                    del partitions[sorted_diff_norms[0]]
                    partitions[diff_norm]=combination
        return list(partitions.values())
