import seaborn as sns
import pandas as pd

original_data = pd.read_csv('results/categorical_results.csv', sep=';',float_precision='round_trip')

df = pd.DataFrame({'PARTITIONERS': original_data['PARTITIONER'],
                   'CLASSIFIERS': original_data['CLASSIFIER'],
                   'ACCURACY':  original_data['ACCURACY']},
                  columns = ['PARTITIONERS','CLASSIFIERS','ACCURACY'])
