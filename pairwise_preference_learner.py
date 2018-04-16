import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
import numbers


def convert_instance(X_encoder, instance):
    test = []
    # X é codificado como One-Hot encoding, entao todas as colunas sao 0 ou 1
    for item in X_encoder:
        # O pd.get_dummies cria colunas do tipo coluna_valor
        label = item.split('_')[0]
        value = item.split('_')[1]
        # crio a instância para classificação no formato do One-Hot encoding
        if isinstance(instance[label], numbers.Number):
            if instance[label] == float(value):
                test.append(1)
            else:
                test.append(0)
        elif instance[label] == value:
            test.append(1)
        else:
            test.append(0)
    return test


class PreferenceRelation():
    def __init__(self, classes, classifier, encoder):
        self.classes = classes
        self.classifier = classifier
        self.encoder = encoder


# lendo dados originais, preciso desse float precision pra ele não arredondar
data = pd.read_csv('data.csv', float_precision='round_trip')
# apenas os dados de sucesso, sem a coluna de erro
data = data.loc[data['erro'] == False].drop('erro', axis=1)
feature = 'num_aligns'

# as classes do meu problema é a feature que quero recomendar
classes = data[feature].unique()
print(classes)
# combino dois a dois das classes
classes_pairs = list(itertools.combinations(classes, 2))
preferences_relations = []
for c1, c2 in classes_pairs:
    # só me importa os dados que pertença a uma das classes
    preferences = pd.concat([data[data[feature] == c1], data[data[feature] == c2]])
    X = preferences.loc[:, preferences.columns != feature]
    y = preferences.loc[:, preferences.columns == feature]
    neigh = KNeighborsClassifier(n_neighbors=3)
    X_ = X.astype(str)
    # One-hot encoding
    X_ = pd.get_dummies(X_)
    # todas as novas colunas após o encoding
    X_encoder = list(X_)
    neigh.fit(X_.values, y.values.ravel())
    #guardo o modelo gerado para esse par
    preferences_relations.append(PreferenceRelation((c1, c2), neigh, X_encoder))

#inicializo os 'votos' zerados
voting_classes = dict.fromkeys(classes,0)
# recupero a probabilidade de predição de cada classificador que utilizou a classe em questão
# e somo a sua 'votação' os 'votos' desse classificador
for class_ in classes:
    for relation in preferences_relations:
        if class_ in relation.classes:
            sorted_classes = sorted(relation.classes)
            instance = convert_instance(relation.encoder, X.iloc[0, :])
            prob_sorted_by_classes = relation.classifier.predict_proba([instance])[0]
            if sorted_classes.index(class_) == 0:
                voting_classes[class_] += prob_sorted_by_classes[0]
            else:
                voting_classes[class_] += prob_sorted_by_classes[1]

#ordeno o dicionario pelos valores, trazendo o rank
rank = sorted(voting_classes.items(), key=lambda x: x[1], reverse=True)
print(rank)

#TODO fazer esse pairwise levar em consideração features recomendadas
#TODO organizar para não replicar código
#TODO generalizar a recomendação para colunos que possam não ser numéricas