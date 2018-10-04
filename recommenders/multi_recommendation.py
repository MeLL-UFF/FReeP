from utils.preference_processor import PreferenceProcessor
import random
from pandas.api.types import is_numeric_dtype
from recommenders.classifier_feature_recommender import ClassifierFeatureRecommender
from recommenders.regressor_feature_recommender import RegressorFeatureRecommender
from sklearn.svm import SVR
from sklearn.svm import SVC
from collections import Counter

class MultiRecommendation():
    ITERATIONS = 5

    def __init__(self, data, partitioner):
        self.data = data
        self.partitioner = partitioner

    def recommend(self, preferences):
        columns_in_preferences = PreferenceProcessor.parameters_in_preferences(
            preferences, self.data.columns.values)
        columns_to_recommend = list(
            set(self.data.columns) - set(columns_in_preferences))

        voters = []
        for i in range(MultiRecommendation.ITERATIONS):
            voters.append(random.sample(
                columns_to_recommend, len(columns_to_recommend)))

        votes = {}
        for column in columns_to_recommend:
            votes[column] = []

        for voter in voters:
            tmp_columns_preferences = columns_in_preferences
            tmp_preferences = preferences
            for column in voter:
                y = self.data[column]
                X = self.data.drop(
                    list(set(columns_to_recommend) - set(tmp_columns_preferences)), axis=1)
                recommender = self.model_type(X, y)
                vote = recommender.recommend(column, tmp_preferences)
                votes[column].append(vote[0])
                preference = self.preference_to_append(y, column, vote[0])
                tmp_preferences.append(preference)
                tmp_columns_preferences.append(column)
        resp = {}
        for param, vts in votes.items():
            elected = Counter(vts).most_common(1)[0][0]
            resp[param] = elected
        return resp

    def model_type(self, X, y):
        if is_numeric_dtype(y):
            return RegressorFeatureRecommender(X, y, self.partitioner,
                                               regressor=SVR())
        else:
            return ClassifierFeatureRecommender(X, y, self.partitioner,
                                                classifier=SVC())

    def preference_to_append(self, y, column, vote_value):
        if is_numeric_dtype(y):
            std = y.std()
            return '( '+str(column) + " <= " + str(vote_value + std) + ' ) | ' \
                   + '( '+ str(column) + " >= " + str(vote_value - std) + ' )'

        else:
            return str(column) + " == '" + str(vote_value) + "'"
