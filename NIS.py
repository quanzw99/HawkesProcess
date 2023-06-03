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
                 learning_rate=0.001, batch_size=1000, save_step=50, epoch_num=20,
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

        if optim == 'SGD':
            self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.delta1])
        elif optim == 'Adam':
            self.opt = Adam(lr=learning_rate, params=[self.node_emb, self.delta1])

        self.loss = torch.FloatTensor()

    def get_dataset(self, data_name):
        with open('./dataset.json', 'r') as dataset_file:
            dataset_data = json.load(dataset_file)
        return dataset_data[data_name]

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, s_h_nodes, s_h_times, s_h_masks,
                t_h_nodes, t_h_times, t_h_masks, n_h_nodes, n_h_times, n_h_masks):

        batch = s_nodes.size()[0]

        s_node_emb = self.node_emb[s_nodes.view(-1)].view(batch, -1)
        t_node_emb = self.node_emb[t_nodes.view(-1)].view(batch, -1)
        n_node_emb = self.node_emb[n_nodes.view(-1)].view(batch, self.neg_size, -1)
        s_h_node_emb = self.node_emb[s_h_nodes.view(-1)].view(batch, self.hist_len, -1)
        t_h_node_emb = self.node_emb[t_h_nodes.view(-1)].view(batch, self.hist_len, -1)
        n_h_node_emb = self.node_emb[n_h_nodes.view(-1)].view(batch, self.neg_size, self.hist_len, -1)

        d1 = torch.sigmoid(self.delta1)
        d2 = 1 - d1

        # calculate p_mu
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()

        # calculate s_hist_emb
        d_s_time = (1. / (1.0 + torch.abs(t_times.unsqueeze(1) - s_h_times))) * s_h_masks
        d_s_time_sum = d_s_time.sum(dim=1).unsqueeze(1) + 1e-6
        s_h_weight = d_s_time / d_s_time_sum # (batch, hist_len)
        s_hist_emb = (s_h_weight.unsqueeze(2) * s_h_node_emb).sum(dim=1)

        # calculate t_hist_emb
        d_t_time = (1. / (1. + torch.abs(t_times.unsqueeze(1) - t_h_times))) * t_h_masks
        d_t_time_sum = d_t_time.sum(dim=1).unsqueeze(1) + 1e-6
        t_h_weight = d_t_time / d_t_time_sum  # (batch, hist_len)
        t_hist_emb = (t_h_weight.unsqueeze(2) * t_h_node_emb).sum(dim=1) # (batch, emb_size)

        # calculate p_lambda
        p_h1 = ((s_hist_emb - t_node_emb) ** 2).sum(dim=1).neg() \
               + ((t_hist_emb - s_node_emb) ** 2).sum(dim=1).neg()
        p_h2 = ((s_hist_emb - t_hist_emb) ** 2).sum(dim=1).neg()
        p_lambda = p_mu + d1 * p_h1 + d2 * p_h2

        # calculate n_mu
        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg() # (batch, neg_size)

        # calculate n_hist_emb
        d_n_time = (1. / (1. + torch.abs(t_times.unsqueeze(-1).unsqueeze(-1) - n_h_times))) * n_h_masks
        d_n_time_sum = d_n_time.sum(dim=2).unsqueeze(2) + 1e-6
        n_h_weight = d_n_time / d_n_time_sum # (batch, neg_size, hist_len)
        n_hist_emb = (n_h_weight.unsqueeze(3) * n_h_node_emb).sum(dim=2) # (batch, neg_size, emb_size)

        # calculate n_lambda
        n_h1 = ((s_hist_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()\
                + ((n_hist_emb - s_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg() # (batch, neg_size)
        n_h2 = ((s_hist_emb.unsqueeze(1) - n_hist_emb) ** 2).sum(dim=2).neg()
        n_lambda = n_mu + d1 * n_h1 + d2 * n_h2

        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, s_h_nodes, s_h_times, s_h_masks,
               t_h_nodes, t_h_times, t_h_masks, n_h_nodes, n_h_times, n_h_masks):

        # calculate p_lambdas and n_lambdas
        p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, s_h_nodes, s_h_times, s_h_masks,
                                            t_h_nodes, t_h_times, t_h_masks, n_h_nodes, n_h_times, n_h_masks)

        loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
            torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)

        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, s_h_nodes, s_h_times, s_h_masks,
               t_h_nodes, t_h_times, t_h_masks, n_h_nodes, n_h_times, n_h_masks):
        self.opt.zero_grad()
        loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, s_h_nodes, s_h_times, s_h_masks,
                              t_h_nodes, t_h_times, t_h_masks, n_h_nodes, n_h_times, n_h_masks)
        loss = loss.sum()
        self.loss += loss.data
        loss.backward()
        self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            # init loss at the beginning of each epoch
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=5)
            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 100 == 0 and i_batch != 0:
                    d1 = torch.sigmoid(self.delta1.data).data.numpy()
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta1:' + str(
                        d1) + '\tdelta2:' + str(1 - d1))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType),
                            sample_batched['target_node'].type(LType),
                            sample_batched['target_time'].type(FType),
                            sample_batched['neg_nodes'].type(LType),
                            sample_batched['s_hist_nodes'].type(LType),
                            sample_batched['s_hist_times'].type(FType),
                            sample_batched['s_hist_masks'].type(FType),
                            sample_batched['t_hist_nodes'].type(LType),
                            sample_batched['t_hist_times'].type(FType),
                            sample_batched['t_hist_masks'].type(FType),
                            sample_batched['n_hists_nodes'].type(LType),
                            sample_batched['n_hists_times'].type(FType),
                            sample_batched['n_hists_masks'].type(FType))

            # print the avg loss for each epoch
            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

        self.save_node_embeddings(self.save_file % (self.epochs))

    def save_node_embeddings(self, file):
        dir = './emb'
        if not os.path.exists(dir):
            os.makedirs(dir)
            print('create the dir...')
        path = dir + '/' + file

        embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()

if __name__ == '__main__':
    # optim = ['SGD', 'Adam']
    nis = NIS(data_name='dblp', optim='Adam')
    nis.train()