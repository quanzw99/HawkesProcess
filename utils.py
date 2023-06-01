from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def format_training_data(emb_file, i2l_file):
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

    X = []
    Y = []
    i2l_list = sorted(i2l.items(), key=lambda x:x[0])
    for (id, label) in i2l_list:
        Y.append(label)
        X.append(i2e[id])

    X = np.stack(X)
    Y = np.array(Y)
    return X, Y


def node_classification(emb_file, i2l_file, test_size):
    X, Y = format_training_data(emb_file, i2l_file)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    return micro_f1, macro_f1


if __name__ == '__main__':
    micro_f1, macro_f1 = node_classification('./emb/dblp_htne_attn_20.emb', './dataset/dblp/nid2label.txt', 0.2)
    print("micro_f1: {:.5f}".format(micro_f1))
    print("macro_f1: {:.5f}".format(macro_f1))