from partitioners.partitioner import Partitioner
from sklearn.decomposition import PCA
import numpy as np


class PCAPartitioner(Partitioner):
    def __init__(self, percentile=50):
        self.percentile = percentile
        super(PCAPartitioner, self).__init__()

    def partition(self, X, y, preferences_columns):
        combinations = super(PCAPartitioner, self).powerset(
            preferences_columns)
        maximum_partitions = int(
            len(X[preferences_columns].columns) * (self.percentile / 100.0))
        partitions = {}
        pca_original = PCA(n_components=2)
        pca_original.fit(X)
        original_singular_values = pca_original.singular_values_
        for combination in combinations:
            partition = X[combination]
            pca_partition = PCA(n_components=2)
            pca_partition.fit(partition)
            partition_singular_values = pca_partition.singular_values_
            singular_values_distance = np.linalg.norm(
                original_singular_values-partition_singular_values)
            if len(partitions) < maximum_partitions:
                partitions[singular_values_distance] = combination
            else:
                sorted_singular_values_distances = sorted(
                    list(partitions.keys()), reverse=True)
                if sorted_singular_values_distances[0] > singular_values_distance:
                    del partitions[sorted_singular_values_distances[0]]
                    partitions[singular_values_distance] = combination
        return list(partitions.values())
