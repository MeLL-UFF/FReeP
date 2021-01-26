#!/usr/local/bin/python3.5

import freep
import numpy as np
import pandas as pd
import random
import csv
import json
import time
import logging
import subprocess
import pydoop.hdfs as hdfs

from freep.utils.preference_processor import is_parameter_in_preferences
from freep.utils.preference_processor import parameters_in_preferences
from freep.utils.preference_processor import parameter_from_encoded_parameter
from freep.utils.preference_processor import get_preferences_for_partition
from freep.utils.encode_decode_processor import encode
from freep.partitioners.commons import all_columns_present
from freep.partitioners import pca
from freep.recommenders import classifier

# sources: https://crs4.github.io/pydoop/tutorial/hdfs_api.html

logging.basicConfig(
    format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)


def run_cmd(args_list):
    """
    run linux commands
    """
    # import subprocess
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = subprocess.Popen(
        args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err

logging.debug('Removendo dados de execução anterior...')
(ret, out, err) = run_cmd(['hadoop', 'fs', '-rm','-r','/user/freep'])

data = pd.read_csv('data.csv', float_precision='round_trip')
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

features = ['model1']

preferences = data.sample(1).to_dict()
true_values = [preferences[feature] for feature in features]
for feature in features:
    del preferences[feature]

model_preferences = []
for key, d in preferences.items():
    for i, value in d.items():
        if type(value) is str:
            model_preferences.append("{} == '{}'".format(
                str(key), str(value)))
        else:
            model_preferences.append(str(key) + ' == ' + str(value))

y = data[feature]
X = data.drop(feature, axis=1)

logging.debug('Target: ' + feature)
logging.debug('Preferences: ' + str(model_preferences))

logging.debug('Generating partitions...')
columns_in_preferences = parameters_in_preferences(
    model_preferences, X.columns.values)
X_encoded, y_encoded, y_encoder = encode(X, y)
partitions_for_recommender = pca.partition(
    X_encoded, y_encoded, columns_in_preferences)

logging.debug('Sending data to HDFS...')

input_path = '/user/freep/input/'
hdfs.mkdir(input_path)
new_preferences = "\n".join(model_preferences)
hdfs.dump(new_preferences, input_path + 'preferences.txt')

partitions = "\n".join(','.join(v.tolist())
                       for v in partitions_for_recommender)
hdfs.dump(partitions, input_path + 'partitions.txt')

hdfs.dump(X.to_csv(index=False), input_path + 'X.csv')
hdfs.dump(y.to_csv(index=False, header=[features[0]]), input_path + 'y.csv')

logging.debug('Running map reduce job...')

(ret, out, err) = run_cmd(['hadoop', 'jar', '/usr/lib/hadoop-mapreduce/hadoop-streaming.jar',
                           '-file', '/app/freep_hadoop/recommender_mapper.py', '-mapper', '/app/freep_hadoop/recommender_mapper.py',
                           '-file', '/app/freep_hadoop/recommender_reducer.py', '-reducer', '/app/freep_hadoop/recommender_reducer.py',
                           '-input', '/user/freep/input/partitions.txt', '-output', '/user/freep/output'])
print('Job status: %s %s %s' % (out.decode('utf-8'), str(ret), str(err)))

ouput_path = '/user/freep/output/'
votes = []
import pdb; pdb.set_trace()
with hdfs.open(ouput_path+'part-00000', 'rt') as fi:
    vote = eval(fi.readline())
    votes.append(vote)
    
print('Recomendação: ' + str(classifier.recomendation(votes)))

(ret, out, err) = run_cmd(['hadoop', 'fs', '-rm','-r','/user/freep'])
