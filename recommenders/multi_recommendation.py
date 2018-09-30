from utils.preference_processor import PreferenceProcessor
import random
from pandas.api.types import is_numeric_dtype
from recommenders.classifier_feature_recommender import ClassifierFeatureRecommender
from recommenders.regressor_feature_recommender import RegressorFeatureRecommender
from sklearn.svm import SVR
from sklearn.svm import SVC


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
                if is_numeric_dtype(y):
                    recommender = RegressorFeatureRecommender(X, y, self.partitioner,
                                                              regressor=SVR())
                else:
                    recommender = ClassifierFeatureRecommender(X, y, self.partitioner,
                                                               classifier=SVC())
                vote = recommender.recommend(column, tmp_preferences)
                #TODO quando o parâmetro que recomendo primeiro é um numeral
                # como é feita a regressão, o filtro horizontal provavelmente vai 
                # ser vazio por não possuir registro com esse valor exato de recomendação
                if vote == None:
                    print('aqui')
                votes[column].append(vote[0])
                if is_numeric_dtype(y):
                    tmp_preferences.append(str(column)+" == "+str(vote[0]))
                else:
                    tmp_preferences.append(
                        str(column)+" == '"+str(vote[0])+"'")
                tmp_columns_preferences.append(column)
            print(votes)
        #TODO votação com todos os parâmetros recomendados