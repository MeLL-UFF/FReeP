#!/usr/local/bin/python3.5
"""reducer.py"""

from operator import itemgetter
import sys
from io import StringIO
import pydoop.hdfs as hdfs
import pandas as pd
from sklearn.svm import SVC

from freep.utils.preference_processor import is_parameter_in_preferences
from freep.utils.preference_processor import parameters_in_preferences
from freep.utils.preference_processor import parameter_from_encoded_parameter
from freep.utils.preference_processor import get_preferences_for_partition
from freep.utils.encode_decode_processor import encode
from freep.partitioners.commons import all_columns_present
from freep.partitioners import pca
from freep.recommenders import classifier

input_path = '/user/freep/input/'

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    if line:
        filename, partition = line.split('\t', 1)
        partition = eval(partition)
        data_text = hdfs.load(filename, mode='rt')
        data = pd.read_csv(StringIO(data_text), sep=",",
                           float_precision='round_trip')
        feature = list(data)[-1]
        y = data[feature]
        X = data.drop(feature, axis=1)
        
        X_encoded, y_encoded, y_encoder = encode(X, y)
        # todas as colunas da partição atual estão no X_encoded?
        if all_columns_present(partition, X_encoded.columns):
            X_partition = pca.vertical_filter(X_encoded, partition)
            print(X_partition)
            vote = classifier.recommender(
                X_partition, y_encoded, feature, partition, SVC(probability=True))
            processed_vote = classifier.process_vote(vote, y_encoder)
            print(processed_vote)
            # votes.append(processed_vote)
