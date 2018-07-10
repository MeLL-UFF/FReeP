from functools import reduce
from partitioners.partitioner import Partitioner

class FullPartitioner(Partitioner):
    def __init__(self):
        super(FullPartitioner, self).__init__()
    
    def partition(self, X, preferences_columns):
        """ Conjunto das partes de todas as colunas das preferências, exceto o vazio"""
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                      preferences_columns, [[]])[1:]