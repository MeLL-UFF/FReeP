import time
import pandas as pd
from experiment_script import ExperimentScript
from graph_generator import GraphGenerator

data = pd.read_csv('sciphy.csv', float_precision='round_trip')
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

categorical_features = ['model1', 'model2']
categorical_result_path = 'results/categorical_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment = ExperimentScript()
experiment.run_classifier(data, categorical_features,
                          5, categorical_result_path)


graph_generator = GraphGenerator()
graph_generator.bar_plot

numerical_features = ['num_aligns', 'length', 'prob1', 'prob2']
numerical_result_path = 'results/numerical_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment.run_regressors(data, numerical_features,
                          5, numerical_result_path)

chart_data = pd.read_csv(
    'results/old/categorical_results.csv', float_precision='round_trip', sep=';')
model1 = data[data['FEATURE'] == 'model1'].drop('FEATURE', axis=1)
