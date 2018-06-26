from partitioners.partitioner import Partitioner
from sklearn.feature_selection import SelectPercentile

class PercentagePartitioner(Partitioner):
    def __init__(self,data, feature, preferences):
        super(PercentagePartitioner, self).__init__(data, feature, preferences)
    
    def partition(self, lst):
        X = self.data[self.data.columns.intersection(list(lst))]
        y = self.data[self.feature]
        feature_selection = SelectPercentile(percentile=50)
        feature_selection.fit(X,y)
        X_ = feature_selection.transform(X)
        pass
        