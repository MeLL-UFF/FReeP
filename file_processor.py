def process_line(line, data, k, feature):
    return feature_line(line, data, k, feature)

def feature_line(line, data, k , feature):
    features = line.split('=')
    if(features[0].strip() == 'FEATURE'):
        if features[1].strip() not in data:
            data[features[1].strip()] = {}
            return (k, features[1].strip())
    else:
        return k_line(line, data, k , feature)

def k_line(line, data, k , feature):
    ks = line.split('=')
    if(ks[0].strip() == 'K'):
        if ks[1].strip() not in data[feature]:
            data[feature][ks[1].strip()] = {}
            return (ks[1].strip(), feature)
    else:
        return knn_line(line, data, k , feature)

def knn_line(line, data, k , feature):
    knns = line.split('\t')
    if(len(knns) == 4):
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
        return rank_line(line, data, k , feature)

def rank_line(line, data, k , feature):
    ranks = line.split('\t')
    if(len(ranks) == 5):
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
print(data)
