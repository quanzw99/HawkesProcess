import json

import torch
from NISDataSet import NISDataSet
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

FType = torch.FloatTensor
LType = torch.LongTensor
torch.set_num_threads(1)

class NIS:
    def __init__(self, data_name, emb_size=128, neg_size=5, hist_len=5, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=20,
                 model_name='nis', optim='SGD'):

        file_path = self.get_dataset(data_name)['edges']
        self.save_file = data_name + '_' + model_name + '_' + optim +'_%d.emb'
        self.model_name = model_name

        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        print('start loading dataset...')
        self.data = NISDataSet(file_path, neg_size, hist_len, directed)
        print('finish loading dataset...')

        # the number of the nodes
        self.node_dim = self.data.get_node_dim()

        self.node_emb = Variable(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
            FType), requires_grad=True)

        self.delta1 = Variable((torch.tensor(1.0)).type(FType), requires_grad=True)
        self.delta2 = Variable((torch.tensor(1.0)).type(FType), requires_grad=True)

        if optim == 'SGD':
            self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.delta1, self.delta2])
        elif optim == 'Adam':
            self.opt = Adam(lr=learning_rate, params=[self.node_emb, self.delta1, self.delta2])

        self.loss = torch.FloatTensor()

    

    def get_dataset(self, data_name):
        with open('./dataset.json', 'r') as dataset_file:
            dataset_data = json.load(dataset_file)
        return dataset_data[data_name]

if __name__ == '__main__':
    # optim = ['SGD', 'Adam']
    nis = NIS(data_name='dblp', optim='Adam')