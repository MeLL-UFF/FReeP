#!/usr/local/bin/python3.5
import os
import sys
import pandas as pd
import pyarrow as pa
from sklearn.svm import SVC
from io import StringIO
import pydoop.hdfs as hdfs
import logging
from datetime import datetime

from freep.utils.preference_processor import is_parameter_in_preferences
from freep.utils.preference_processor import parameters_in_preferences
from freep.utils.preference_processor import parameter_from_encoded_parameter
from freep.utils.preference_processor import get_preferences_for_partition
from freep.utils.encode_decode_processor import encode
from freep.partitioners.commons import all_columns_present
from freep.partitioners import pca


logging.basicConfig(
    format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

input_path = '/user/freep/input/'

logging.debug('Loading partitioners...')
X_TEXT = hdfs.load(input_path+'X.csv', mode='rt')
y_TEXT = hdfs.load(input_path+'y.csv', mode='rt')
preferences = hdfs.load(input_path+'preferences.txt', mode='rt')

preferences = preferences.split('\n')
X = pd.read_csv(StringIO(X_TEXT), sep=",", float_precision='round_trip')
y = pd.read_csv(StringIO(y_TEXT), sep=",", float_precision='round_trip')

logging.debug('MAPPER')
for line in sys.stdin:
    line = line.strip()
    if line:
        partition = line.split(',')

        preferences_for_partition = get_preferences_for_partition(X, partition, preferences)
        # aplicar o filtro das preferencias no X e y originais
        X_, y_, weights_ = pca.horizontal_filter(
            X, y, preferences_for_partition)
        if X_.empty == False:
            now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
            filename = input_path + 'data_'+now+'.csv'
            data = pd.concat([X_, y_], axis=1)
            hdfs.dump(data.to_csv(index=False), filename)
            print("%s\t%s" % (filename, partition))
