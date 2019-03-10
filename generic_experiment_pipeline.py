import time
import pandas as pd
import generic_experiment_script as experiment
# from graph_generator import GraphGenerator

sciphy = pd.read_csv('sciphy.csv', float_precision='round_trip')
sciphy = sciphy[~sciphy['erro']].copy().drop('erro', axis=1).reset_index(drop=True)
result_path = 'results/sciphy/generic_results' + \
    time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

experiment.run(sciphy, result_path)

######################################################################################
######################################################################################

# montage = pd.read_csv('montage.csv', float_precision='round_trip')
# columns = ['cntr', 'ra', 'dec', 'cra', 'cdec', 'crval1', 'crval2', 'crota2']
# montage = montage[columns]

# result_path = 'results/montage/generic_results' + \
#     time.strftime('%a, %d %b %Y %H:%M:%S ') + '.csv'

# experiment.run(montage, result_path)