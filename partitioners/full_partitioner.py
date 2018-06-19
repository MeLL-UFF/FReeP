from functools import reduce
from partitioner import Partitioner

class FullPartitioner(Partitioner):
    def __init__(self,data, preferences):
        super(FullPartitioner, self).__init__(data, preferences)
    
    def partition(self, lst, data):
        """ Conjunto das partes de todas as colunas das preferências, exceto o vazio"""
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                      lst, [[]])[1:]