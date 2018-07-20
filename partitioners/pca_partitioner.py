from partitioners.partitioner import Partitioner
from sklearn.decomposition import PCA
import numpy as np
from preprocessors.encoding_processor import EncodingProcessor
from utils.preference_processor import PreferenceProcessor


class PCAPartitioner(Partitioner):
    def __init__(self, percentile=50):
        self.percentile = percentile
        super(PCAPartitioner, self).__init__()

    def partition(self, X, y, preferences_columns, preferences_parameters):
        combinations = super(PCAPartitioner, self).powerset(
            preferences_columns)
        maximum_partitions = int(
            len(X[preferences_parameters].columns) * (self.percentile / 100.0))
        partitions = {}
        pca_original = PCA(n_components=2)

        self.preprocessor = EncodingProcessor()
        X_, y_ = self.preprocessor.encode(
            X, y)

        pca_original.fit(X_)
        original_singular_values = pca_original.singular_values_
        for combination in combinations:
            partition = X
            for item in combination:
                partition = X[eval(PreferenceProcessor.preference_for_eval(
                    item, X.columns.values))]
            # partition = X[eval(PreferenceProcessor.preference_for_eval(
            #     combination, X.columns.values))]
            partition, y__ = self.preprocessor.encode(
                partition, y)
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
