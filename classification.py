from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import lil_matrix
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')


def format_training_data_for_dnrl(emb_file, i2l_file):
    i2l = dict()
    with open(i2l_file, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            i2l[n_id] = l_id

    i2e = dict()
    with open(emb_file, 'r') as reader:
        reader.readline()
        node_id = 0
        for line in reader:
            if node_id in i2l:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                i2e[node_id] = embeds
            node_id += 1

    Y = []
    X = []
    i2l_list = sorted(i2l.items(), key=lambda x:x[0])
    for (id, label) in i2l_list:
        Y.append(label)
        X.append(i2e[id])

    X = np.stack(X)
    return X, Y


def lr_classification(X, Y, cv):
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, Y, cv=cv, scoring='f1_weighted', n_jobs=8)
    return scores


if __name__ == '__main__':
    X, Y = format_training_data_for_dnrl('./emb/dblp_htne_attn_1.emb', './dataset/nid2label.txt')
    print(lr_classification(X, Y, cv=5))
