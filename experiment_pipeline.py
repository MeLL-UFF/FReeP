import time
import pandas as pd
from experiment_script import ExperimentScript
from graph_generator import GraphGenerator

data = pd.read_csv('sciphy.csv', float_precision='round_trip')
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

categorical_features = ['model1', 'model2']
categorical_result_path = 'results/sciphy/categorical_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment = ExperimentScript()
experiment.run_classifier(data, categorical_features,
                          5, categorical_result_path)

categorical_time_graph_path = 'results/sciphy/categorical_time_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(categorical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, categorical_time_graph_path, "PARTITIONER", "TIME", "CLASSIFIER", "TIME")

categorical_accuracy_graph_path = 'results/sciphy/categorical_accuracy_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(categorical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, categorical_accuracy_graph_path, "PARTITIONER", "ACCURACY", "CLASSIFIER", "ACCURACY")

#
# ################################################################
numerical_features = ['num_aligns', 'length', 'prob1', 'prob2']
numerical_result_path = 'results/sciphy/numerical_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment.run_regressors(data, numerical_features,
                          5, numerical_result_path)
numerical_time_graph_path = 'results/sciphy/numerical_time_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(numerical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, numerical_time_graph_path, "PARTITIONER", "TIME", "REGRESSOR", "TIME")

numerical_mse_graph_path = 'results/sciphy/numerical_mse_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(numerical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, numerical_mse_graph_path, "PARTITIONER", "MSE", "REGRESSOR", "MSE")


######################################################################################
######################################################################################

data = pd.read_csv('montage.csv', float_precision='round_trip')
columns = ['cntr', 'ra', 'dec', 'cra', 'cdec', 'crval1', 'crval2', 'crota2']
data = data[columns]

categorical_features = ['cra', 'cdec']
categorical_result_path = 'results/montage/categorical_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment = ExperimentScript()
experiment.run_classifier(data, categorical_features,
                          5, categorical_result_path)

categorical_time_graph_path = 'results/montage/categorical_time_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(categorical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, categorical_time_graph_path, "PARTITIONER", "TIME", "CLASSIFIER", "TIME")

categorical_accuracy_graph_path = 'results/montage/categorical_accuracy_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(categorical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, categorical_accuracy_graph_path, "PARTITIONER", "ACCURACY", "CLASSIFIER", "ACCURACY")


################################################################
numerical_features = ['cntr', 'ra', 'dec', 'crval1', 'crval2', 'crota2']
numerical_result_path = 'results/montage/numerical_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment.run_regressors(data, numerical_features,
                          5, numerical_result_path)
numerical_time_graph_path = 'results/montage/numerical_time_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(numerical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, numerical_time_graph_path, "PARTITIONER", "TIME", "REGRESSOR", "TIME")

numerical_mse_graph_path = 'results/montage/numerical_mse_graph' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.pdf'
graph_data = pd.read_csv(numerical_result_path, float_precision='round_trip', sep=';')
graph_generator = GraphGenerator()
graph_generator.bar_graph(graph_data, numerical_mse_graph_path, "PARTITIONER", "MSE", "REGRESSOR", "MSE")
