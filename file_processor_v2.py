import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randint


def process_line(line, data, k, feature):
    return feature_line(line, data, k, feature)


def feature_line(line, data, k, feature):
    features = line.split('=')
    if (features[0].strip() == 'FEATURE'):
        if features[1].strip() not in data:
            data[features[1].strip()] = {}
            return (k, features[1].strip())
    else:
        return k_line(line, data, k, feature)


def k_line(line, data, k, feature):
    ks = line.split('=')
    if (ks[0].strip() == 'K'):
        if ks[1].strip() not in data[feature]:
            data[feature][ks[1].strip()] = {}
            return (ks[1].strip(), feature)
    else:
        return knn_line(line, data, k, feature)


def knn_line(line, data, k, feature):
    knns = line.split('\t')
    if (len(knns) == 4):
        if 'KNN' not in data[feature][k]:
            data[feature][k]['KNN'] = {}
        if 'ACURACIA' not in data[feature][k]['KNN']:
            data[feature][k]['KNN']['ACURACIA'] = []
        data[feature][k]['KNN']['ACURACIA'].append(knns[1].strip())
        if 'PRECISAO' not in data[feature][k]['KNN']:
            data[feature][k]['KNN']['PRECISAO'] = []
        data[feature][k]['KNN']['PRECISAO'].append(knns[2].strip())
        if 'RECALL' not in data[feature][k]['KNN']:
            data[feature][k]['KNN']['RECALL'] = []
        data[feature][k]['KNN']['RECALL'].append(knns[3].strip())
        return (k, feature)
    else:
        return rank_line(line, data, k, feature)


def rank_line(line, data, k, feature):
    ranks = line.split('\t')
    if (len(ranks) == 5):
        if 'RANK' not in data[feature][k]:
            data[feature][k]['RANK'] = {}
        if 'ACURACIA' not in data[feature][k]['RANK']:
            data[feature][k]['RANK']['ACURACIA'] = []
        data[feature][k]['RANK']['ACURACIA'].append(ranks[2].strip())
        if 'PRECISAO' not in data[feature][k]['RANK']:
            data[feature][k]['RANK']['PRECISAO'] = []
        data[feature][k]['RANK']['PRECISAO'].append(ranks[3].strip())
        if 'RECALL' not in data[feature][k]['RANK']:
            data[feature][k]['RANK']['RECALL'] = []
        data[feature][k]['RANK']['RECALL'].append(ranks[4].strip())
    return (k, feature)


data = {}
with open('only_metrics.txt') as f:
    k = -1
    feature = ""
    for line in f:
        line = line.rstrip('\n')
        result = process_line(line, data, k, feature)
        k = result[0]
        feature = result[1]

acc_data = [['Parâmetro', 'K', 'Modelo', 'Acurácia', 'STD']]
std = []
precision_data = [['Parâmetro', 'K', 'Modelo', 'Precisão']]

for param, value in data.items():
    for k, model in value.items():
        for model_type, metrics in model.items():
            acc_values = np.array(data[param][k][model_type]['ACURACIA']).astype(np.float)
            mean_acc = np.mean(acc_values)
            std.append(np.std(acc_values))
            acc_data.append([param, k, model_type, mean_acc, np.std(acc_values)])
            precision_values = np.array(data[param][k][model_type]['PRECISAO']).astype(np.float)
            mean_precision = np.mean(precision_values)
            precision_data.append([param, k, model_type, mean_precision])

acc_frame = pd.DataFrame(data=np.array(acc_data)[1:][:], columns=np.array(acc_data)[0, :])
acc_frame['Acurácia'] = acc_frame['Acurácia'].astype(float)
acc_frame['K'] = acc_frame['K'].astype(int)

bars = []
values = []
std = []
for param in acc_frame['Parâmetro'].unique():
    part = acc_frame[acc_frame['Parâmetro'] == param][acc_frame['Modelo'] == 'RANK'].sort_values(by='K')
    tmp_bars = part[['K', 'Acurácia']]
    tmp_std = part[['STD']]
    bars.append(tmp_bars['K'].values)
    values.append(tmp_bars['Acurácia'].values)
    std.append(tmp_std['STD'].values)

# width of the bars
barWidth = 0.3

colors =['#37AB65', '#3DF735', '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#F6CC1D']

r1 = np.arange(len(values))
for idx,value in enumerate(values):
    err = std[idx]
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, value, width=barWidth, color=colors[idx], edgecolor='black', yerr=err.astype(float), label=''+str(idx))
    r1 = r2

plt.xticks([r + barWidth for r in range(6)], ['cond_A', 'cond_B', 'cond_C','cond_D','cond_E','cond_F'])
plt.ylabel('height')
# plt.legend()

# Show graphic
plt.show()
# Choose the height of the blue bars
# bars1 = [10, 9, 2]
#
# # Choose the height of the cyan bars
# bars2 = [10.8, 9.5, 4.5]
#
# # Choose the height of the error bars (bars1)
# yer1 = [0.5, 0.4, 0.5]
#
# # Choose the height of the error bars (bars2)
# yer2 = [1, 0.7, 1]
#
# # The x position of bars
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
#
# # Create blue bars
# plt.bar(r1, bars1, width=barWidth, color='blue', edgecolor='black', yerr=yer1, capsize=7, label='poacee')
#
# # Create cyan bars
# plt.bar(r2, bars2, width=barWidth, color='cyan', edgecolor='black', yerr=yer2, capsize=7, label='sorgho')

# general layout
# plt.xticks([r + barWidth for r in range(len(bars1))], ['cond_A', 'cond_B', 'cond_C'])
# plt.ylabel('height')
# plt.legend()
#
# # Show graphic
# plt.show()

# g = sns.factorplot(data=acc_frame, x="K", y="Acurácia", hue='Parâmetro', col="Modelo", ci="sd", kind="bar",
#                    palette="muted", legend=False)

# g = sns.factorplot(data=acc_frame[acc_frame['Modelo'] == 'KNN'], x="K", y="Acurácia", hue='Parâmetro', col="Modelo", ci="sd", kind="bar",
#                    palette="muted", legend=False)

# def errplot(x, y, yerr, **kwargs):
#     ax = plt.gca()
#     data = kwargs.pop("data")
#     data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
#
# g.map_dataframe(errplot, "K", "Acurácia", "STD")
#
# plt.legend(title="Parâmetros",fontsize=3,loc='upper center', bbox_to_anchor=(0.9, 1.1),prop={'size': 6}) #horizontal legend bottom
#
# # Show plot
# plt.show()
#
# precision_frame = pd.DataFrame(data=np.array(precision_data)[1:][:], columns=np.array(precision_data)[0, :])
# precision_frame['Precisão'] = precision_frame['Precisão'].astype(float)
#
# g = sns.factorplot(data=precision_frame, x="K", y="Precisão", hue='Parâmetro', col="Modelo", ci="sd", kind="bar",
#                    palette="muted", legend=False)
#
# plt.legend(title="Parâmetros",fontsize=3,loc='upper center', bbox_to_anchor=(0.9, 1.1),prop={'size': 6}) #horizontal legend bottom
#
# plt.show()

# test_data = np.array(data['length']['3']['KNN']['ACURACIA']).astype(np.float)
# test_mean_accuracy = np.mean(test_data)
# test_std_deviation = np.std(test_data)

# test = [['K','Acurácia','nome']]
# for idx, val in enumerate(test_data):
#     test.append([3,val, 1])
#
# test_ = pd.DataFrame(data=np.array(test)[1:][:],columns=np.array(test)[0,:] )
# test_['Acurácia'] = test_['Acurácia'].astype(float)
#
# # Load data
# titanic = sns.load_dataset("titanic")

# Set up a factorplot
# g = sns.factorplot("class", "survived", "sex", data=titanic, kind="bar", palette="muted", legend=True)

# g = sns.factorplot(x="K", y="Acurácia", hue='nome' ,data=test_, kind="bar", palette="muted", legend=True)
#
# # Show plot
# plt.show()
