# source activate py35
import pandas as pd
from recommenders.feature_recommender import FeatureRecommender
from classifiers.custom_knn import CustomKNN
from collections import Counter


class CustomKNNFeatureRecommender(FeatureRecommender):

    def __init__(self, data, partitioner, weights=[]):
        super(CustomKNNFeatureRecommender, self).__init__(data, partitioner, weights)

    def recommender(self, data, feature, preferences, weights=[]):
        # X = todas as colunas menos a última, Y= última
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # forço todas as colunas serem string
        X = X.astype(str)
        # One-hot encoding
        X = pd.get_dummies(X, prefix_sep="_dummy_")
        # todas as novas colunas após o encoding
        X_encoder = list(X)
        X['class']=y
        neigh = CustomKNN(X.values,k=FeatureRecommender.NEIGHBORS, weights=weights)
        test = []
        # X é codificado como One-Hot encoding, entao todas as colunas sao 0 ou 1
        for item in X_encoder:
            # O pd.get_dummies cria colunas do tipo coluna_valor
            label = item.split('_dummy_')[0]
            value = item.split('_dummy_')[1]
            # crio a instância para classificação no formato do One-Hot encoding
            if label in preferences:
                if preferences[label] == value or preferences[label] == float(value):
                    test.append(1)
                else:
                    test.append(0)
            else:
                test.append(0)
        return neigh.predict([test])[0]

    def recomendation(self, votes):
        c = Counter(votes)
        resp = c.most_common(1)[0][0]
        confidence = c.most_common(1)[0][1] / float(len(votes))
        return (resp, confidence)

    def process_vote(self, votes):
        if self.label_encoder is None:
            return votes
        else:
            decode = self.label_encoder.inverse_transform(votes[0][0])
            return [(decode,votes[0][1])]
