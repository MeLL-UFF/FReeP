from pyhive import hive
from TCLIService.ttypes import TOperationState
import freep
import numpy as np
import pandas as pd
import random
import csv
import json
import time
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

from freep.utils.log import setup_custom_logger
logging = setup_custom_logger('root')

DROP_DATABASE = "DROP SCHEMA IF EXISTS freep CASCADE"
CREATE_DATABASE = 'CREATE SCHEMA IF NOT EXISTS freep'
TRANSFER_QUERY = 'INSERT INTO TABLE freep.orc_table SELECT * FROM freep.csv_table'
INPUT_PATH = '/user/freep/input/'

conn = hive.Connection(host="localhost", port=10000, username="root")
cursor = conn.cursor()

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

def generate_orc_table(header):
	external_table = "CREATE TABLE freep.orc_table "+\
"({}) ".format(','.join(["%s STRING" % x for x in header]))+\
"ROW FORMAT DELIMITED "+\
"STORED AS ORC"
	return external_table

def generate_external_table(header, location):
	external_table = "CREATE EXTERNAL TABLE IF NOT EXISTS freep.csv_table " + \
"({}) ".format(','.join(["%s STRING" % x for x in header])) + \
"ROW FORMAT DELIMITED " + \
"FIELDS TERMINATED BY ',' " + \
"STORED AS TEXTFILE " + \
"LOCATION '{}' ".format(location) +\
"TBLPROPERTIES (\"skip.header.line.count\"=\"1\")"
	return external_table 

def load_data_to_hive(data_path):
    logging.debug('Removendo dados de execução anterior...')
    (ret, out, err) = run_cmd(['hadoop', 'fs', '-rm','-r','/user/freep'])
    logging.debug('Recriando diretórios...')
    hdfs.mkdir(INPUT_PATH)
    logging.debug('Copiando dados para HDFS...')
    hdfs.put(data_path, INPUT_PATH)

    location = 'hdfs://quickstart.cloudera:8020'+INPUT_PATH
    with open(data_path,'r') as opened_in_file:
        reader = csv.DictReader(opened_in_file)
        header = reader.fieldnames

    logging.debug('Criando base no HIVE...')
    cursor.execute(DROP_DATABASE)
    cursor.execute(CREATE_DATABASE)
    logging.debug('Gerando tabela externa...')
    cursor.execute(generate_external_table(header, location))
    logging.debug('Gerando tabela ORC...')
    cursor.execute(generate_orc_table(header))
    logging.debug('Realizando consulta...')
    cursor.execute('SELECT * FROM freep.csv_table LIMIT 10')
    logging.debug(cursor.fetchone())

    logging.debug('Transferindo dados...')
    cursor.execute(TRANSFER_QUERY)
    logging.debug('Realizando consulta...')
    cursor.execute('SELECT * FROM freep.orc_table LIMIT 10')
    logging.debug(cursor.fetchall())
    logging.debug('Finalizando...')

load_data_to_hive('data.csv')

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

partitions_filenames = []

for idx, partition in enumerate(partitions_for_recommender):
    # preferences_for_partition = get_preferences_for_partition(
    #                                 X, partition, preferences)
    preferences_for_partition = get_preferences_for_partition(X, partition, preferences)
    preferences_filter = []
    for preference in preferences_for_partition:
        preferences_filter.append("%s == '%s'" % (preference, list(preferences[preference].values())[0]))
    
    query = ' AND '.join(preferences_filter)
    columns = ', '.join(preferences_for_partition)
    cursor.execute('SELECT {} \
                    FROM freep.csv_table\
                    WHERE {}'.format(columns, query))
    
    values = []
    while True:
        row = cursor.fetchone()
        if row == None:
            break
        values.append(row)
    df = pd.DataFrame(columns=preferences_for_partition, data=values)
    filename = INPUT_PATH + 'partition_'+str(idx)+'.csv'
    hdfs.dump(df.to_csv(index=False), filename)
    
    partitions_filenames.append("%s\t%s" % (filename, partition))

cursor.close()
conn.close()

partitions = "\n".join(partitions_filenames)
hdfs.dump(partitions, INPUT_PATH + 'partitions.txt')

(ret, out, err) = run_cmd(['hadoop', 'jar', '/usr/lib/hadoop-mapreduce/hadoop-streaming.jar',
                           '-file', '/app/hadoop_study/recommender_mapper.py', 
                           '-mapper', '/app/hadoop_study/recommender_mapper.py',
                           '-file', '/app/hadoop_study/recommender_reducer.py', 
                           '-reducer', '/app/hadoop_study/recommender_reducer.py',
                           '-input', INPUT_PATH + 'partitions.txt', '-output', '/user/freep/output'])
logging.debug('Job status: %s %s %s' % (out.decode('utf-8'), str(ret), str(err)))

ouput_path = '/user/freep/output/'

with hdfs.open(ouput_path+'part-00000', 'rt') as fi:
    print(fi.readline())