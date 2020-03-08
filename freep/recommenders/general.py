import random
from collections import Counter

from sklearn.svm import SVR
from pandas.api.types import is_numeric_dtype
from sklearn.svm import SVC

from ..utils.preference_processor import is_parameter_in_preferences
from ..utils.preference_processor import parameters_in_preferences
from ..utils.preference_processor import preference_to_append
from ..utils.preference_processor import parameter_from_encoded_parameter
from ..utils.encode_decode_processor import encode
from ..partitioners.commons import all_columns_present

from ..recommenders import classifier
from ..recommenders import regressor


def recommend(data,    preferences,    partitioner,
              regressor_model=SVR(),
              classifier_model=SVC(probability=True),
              min_neighbors=3,
              max_iter=5,
              ):
    columns_in_preferences = parameters_in_preferences(
        preferences, data.columns.values
    )
    columns_to_recommend = list(
        set(data.columns) - set(columns_in_preferences))

    voters = []
    for i in range(max_iter):
        voters.append(random.sample(
            columns_to_recommend, len(columns_to_recommend)))

    votes = {}
    for column in columns_to_recommend:
        votes[column] = []

    for voter in voters:
        tmp_columns_preferences = columns_in_preferences
        tmp_preferences = preferences
        for column in voter:
            y = data[column]
            X = data.drop(
                list(
                    set(columns_to_recommend) - set(tmp_columns_preferences)
                ), axis=1)

            if is_numeric_dtype(y):
                vote = regressor.recommend(
                    X, y, column, tmp_preferences, partitioner,
                    regressor_model, min_neighbors)
            else:
                vote = classifier.recommend(
                    X, y, column, tmp_preferences, partitioner,
                    classifier_model, min_neighbors)
            
            if vote:
                votes[column].append(vote[0])
                preference = preference_to_append(y, column, vote[0])
                tmp_preferences.append(preference)
                tmp_columns_preferences.append(column)
    resp = {}
    for param, vts in votes.items():
        if len(vts)>0:
            elected = Counter(vts).most_common(1)[0][0]
            resp[param] = elected
        else:
            resp[param] = None
    return resp
