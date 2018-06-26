# source activate py35
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from recommenders.feature_recommender import FeatureRecommender
from collections import Counter


class KNNFeatureRecommender(FeatureRecommender):
    def __init__(self, X, y, partitioner, weights=[], neighbors= FeatureRecommender.NEIGHBORS):
        super(KNNFeatureRecommender, self).__init__(X, y, partitioner, weights, neighbors)

    def recommender(self, data, feature, preferences, weights):
        # X = todas as colunas menos a última, Y= última
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # forço todas as colunas serem string
        if isinstance(y.values[0], float):
            y = y.astype(str)
        X = X.astype(str)
        # One-hot encoding
        X = pd.get_dummies(X, prefix_sep="_dummy_")
        # todas as novas colunas após o encoding
        X_encoder = list(X)
        neigh = KNeighborsClassifier(n_neighbors=self.neighbors)
        neigh.fit(X.values, y.values)
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
        pred = neigh.predict([test])[0]
        prob = max(neigh.predict_proba([test])[0])
        return [(pred,prob)]

    def recomendation(self, votes):
        l = [vote[0] for candidates in votes for vote in candidates]
        c = Counter(l)
        resp = c.most_common(1)[0][0]
        confidence = c.most_common(1)[0][1] / float(len(votes))
        return (resp, confidence)

    def process_vote(self, votes):
        if self.label_encoder is None:
            return votes
        else:
            decode = self.label_encoder.inverse_transform(votes[0][0])
            return [(decode,votes[0][1])]