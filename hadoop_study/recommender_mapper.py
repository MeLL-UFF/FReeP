#! /usr/local/bin/python3.5
import os
import sys
import pandas as pd
import pyarrow as pa
from sklearn.svm import SVC
from io import StringIO
import pydoop.hdfs as hdfs
from datetime import datetime
import logging

from freep.utils.preference_processor import is_parameter_in_preferences
from freep.utils.preference_processor import parameters_in_preferences
from freep.utils.preference_processor import parameter_from_encoded_parameter
from freep.utils.preference_processor import get_preferences_for_partition
from freep.utils.encode_decode_processor import encode
from freep.partitioners.commons import all_columns_present
from freep.partitioners import pca
from freep.recommenders import classifier

from constants import Constants

logging.basicConfig(
    format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)


logging.debug('MAPPER')
# for line in sys.stdin:
f = open('/app/partitions.txt')
for line in f.readlines() :

    line = line.strip()
    print(line)
    if line:
        filename, partition = line.split('\t', 1)
        partition = partition.split(',')
        x_text = hdfs.load(Constants.HDFS_INPUT_PATH+ 'X_'+filename, mode='rt')
        X = pd.read_csv(StringIO(x_text), sep=",",
                            float_precision='round_trip')
        y_text = hdfs.load(Constants.HDFS_INPUT_PATH+'y_'+filename, mode='rt')
        y = pd.read_csv(StringIO(y_text), sep=",",
                            float_precision='round_trip')
        feature = y.columns[0]
        X_encoded, y_encoded, y_encoder = encode(X, y[feature])
        # todas as colunas da partição atual estão no X_encoded?
        if all_columns_present(partition, X_encoded.columns):
            import pdb; pdb.set_trace()
            # X_partition = pca.vertical_filter(X_encoded, partition)
            vote = classifier.recommender(X, y_encoded, feature, partition, SVC(probability=True))
            print(vote)
            processed_vote = classifier.process_vote(vote, y_encoder)
            print(processed_vote)
    print('Next')
