import json

import torch
from HTNEDataSet import HTNEDataSet
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

class HP:
    def __init__(self, data_name, emb_size=128, neg_size=5, hist_len=5, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=20,
                 model_name='htne', optim='SGD'):

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
        self.data = HTNEDataSet(file_path, neg_size, hist_len, directed)
        print('finish loading dataset...')

        # the number of the nodes
        self.node_dim = self.data.get_node_dim()

        self.node_emb = Variable(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
            FType), requires_grad=True)

        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

        if optim == 'SGD':
            self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.delta])
        elif optim == 'Adam':
            self.opt = Adam(lr=learning_rate, params=[self.node_emb, self.delta])

        self.loss = torch.FloatTensor()

    def get_dataset(self, data_name):
        with open('./dataset.json', 'r') as dataset_file:
            dataset_data = json.load(dataset_file)
        return dataset_data[data_name]

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        # the size of the current batch
        batch = s_nodes.size()[0]

        # get the embedding by index
        s_node_emb = self.node_emb[s_nodes.view(-1)].view(batch, -1)
        t_node_emb = self.node_emb[t_nodes.view(-1)].view(batch, -1)
        h_node_emb = self.node_emb[h_nodes.view(-1)].view(batch, self.hist_len, -1)

        if self.model_name == 'htne':
            att = torch.ones((batch, self.hist_len))
        elif self.model_name == 'htne_attn':
            att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta[s_nodes.view(-1)].unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * d_time) * h_time_mask).sum(dim=1)

        n_node_emb = self.node_emb[n_nodes.view(-1)].view(batch, self.neg_size, -1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * d_time).unsqueeze(2)) * (
                h_time_mask.unsqueeze(2))).sum(dim=1)
        return p_lambda, n_lambda

    def bi_forward(self, s_nodes, t_nodes, n_nodes):
        # the size of the current batch
        batch = s_nodes.size()[0]

        # get the embedding by index
        s_node_emb = self.node_emb[s_nodes.view(-1)].view(batch, -1)
        t_node_emb = self.node_emb[t_nodes.view(-1)].view(batch, -1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()

        n_node_emb = self.node_emb[n_nodes.view(-1)].view(batch, self.neg_size, -1)
        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        return p_mu, n_mu

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):

        if self.model_name == 'bi':
            p_lambdas, n_lambdas = self.bi_forward(s_nodes, t_nodes, n_nodes)
        elif self.model_name == 'htne' or self.model_name == 'htne_attn':
            p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times,
                                                h_time_mask)

        loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
            torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)

        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        self.opt.zero_grad()
        loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
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
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType),
                            sample_batched['target_node'].type(LType),
                            sample_batched['target_time'].type(FType),
                            sample_batched['neg_nodes'].type(LType),
                            sample_batched['history_nodes'].type(LType),
                            sample_batched['history_times'].type(FType),
                            sample_batched['history_masks'].type(FType))

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
    # model_name = ['htne_attn', 'htne', 'bi']
    # optim = ['SGD', 'Adam']
    hp = HP(data_name='dblp', model_name='bi', optim='Adam')
    hp.train()
