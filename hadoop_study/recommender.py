import freep
import numpy as np
import pandas as pd
import random
import csv
import json
import time
import pydoop.hdfs as hdfs

from freep.utils.preference_processor import is_parameter_in_preferences
from freep.utils.preference_processor import parameters_in_preferences
from freep.utils.preference_processor import parameter_from_encoded_parameter
from freep.utils.preference_processor import get_preferences_for_partition
from freep.utils.encode_decode_processor import encode
from freep.partitioners.commons import all_columns_present
from freep.partitioners import pca
from freep.recommenders import classifier

from hive_processor import HiveProcessor
from constants import Constants

from freep.utils.commands import run_cmd
from freep.utils.log import setup_custom_logger
logging = setup_custom_logger('root')

def generate_partitions(preferences, X):
    columns_in_preferences = parameters_in_preferences(
        preferences, X.columns.values)
    X_encoded, y_encoded, y_encoder = encode(X, y)
    partitions = pca.partition(
        X_encoded, y_encoded, columns_in_preferences)
    return partitions

def generate_fake_preferences(data, features):
    sample = data.sample(1).to_dict()
    true_values = [sample[feature] for feature in features]
    for feature in features:
        del sample[feature]

    preferences = []
    for key, d in sample.items():
        for i, value in d.items():
            if type(value) is str:
                preferences.append("{} == '{}'".format(
                    str(key), str(value)))
            else:
                preferences.append(str(key) + ' == ' + str(value))
    return sample, preferences, true_values

input_data_path = 'data.csv'

logging.debug('Removendo dados de execução anterior...')
(ret, out, err) = run_cmd(['hadoop', 'fs', '-rm','-r','/user/freep'])
logging.debug('Recriando diretórios...')
hdfs.mkdir(Constants.HDFS_INPUT_PATH)
logging.debug('Copiando dados para HDFS...')
hdfs.put(input_data_path, Constants.HDFS_INPUT_PATH)

with open(input_data_path,'r') as opened_in_file:
    reader = csv.DictReader(opened_in_file)
    header = reader.fieldnames

hive = HiveProcessor()
hive.load_data_to_hive(header)

data = pd.read_csv('data.csv', float_precision='round_trip')
data = data[~data['erro']].copy().drop('erro', axis=1).reset_index(drop=True)

features = ['model1']
feature = features[0]
y = data[feature]
X = data.drop(feature, axis=1)
logging.debug('Target: ' + feature)

sample, preferences, true_values = generate_fake_preferences(data, features)
logging.debug('Preferences: ' + str(preferences))
logging.debug('Generating partitions...')
partitions = generate_partitions(preferences, X)
partitions_filenames = []

for idx, partition in enumerate(partitions):
    preferences_for_partition = get_preferences_for_partition(
                                    X, partition, sample)
    preferences_filter = []
    for preference in preferences_for_partition:
        preferences_filter.append("%s == '%s'" % (preference, 
                                                  list(sample[preference].values())[0]))
    #############horizontal filter##############
    query = ' AND '.join(preferences_filter)
    columns = ', '.join(preferences_for_partition)
    values = hive.query_orc_table(columns, query)
    #############################################
    
    df = pd.DataFrame(columns=preferences_for_partition, data=values)
    y_values = data[feature]
    y_ = pd.DataFrame(columns=[feature], data=y_values)
    X_ = data.drop(feature, axis=1)
    suffix_name = str(idx)+'.csv'
    x_name = Constants.HDFS_INPUT_PATH + 'X_'+suffix_name
    hdfs.dump(X_.to_csv(index=False), x_name)
    y_name = Constants.HDFS_INPUT_PATH + 'y_'+suffix_name
    hdfs.dump(y_.to_csv(index=False), y_name)
    partitions_filenames.append("%s\t%s" % (suffix_name, ','.join(partition.tolist())))

hive.close()

partitions = "\n".join(partitions_filenames)
hdfs.dump(partitions, Constants.HDFS_INPUT_PATH + 'partitions.txt')
import pdb; pdb.set_trace()

(ret, out, err) = run_cmd(['hadoop', 'jar', '/usr/lib/hadoop-mapreduce/hadoop-streaming.jar',
                           '-file', '/app/hadoop_study/recommender_mapper.py', 
                           '-mapper', '/app/hadoop_study/recommender_mapper.py',
                           '-file', '/app/hadoop_study/recommender_reducer.py', 
                           '-reducer', '/app/hadoop_study/recommender_reducer.py',
                           '-input', Constants.HDFS_INPUT_PATH + 'partitions.txt', '-output', '/user/freep/output'])
logging.debug('Job status: %s %s %s' % (out.decode('utf-8'), str(ret), str(err)))


with hdfs.open(Constants.HDFS_OUTPUT_PATH+'part-00000', 'rt') as fout:
    for cnt, line in enumerate(fout):
       print("Line {}: {}".format(cnt, line))