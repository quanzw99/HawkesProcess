import torch
from HTNEDataSet import HTNEDataSet
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
import sys

FType = torch.FloatTensor
LType = torch.LongTensor
torch.set_num_threads(1)

class HTNE_a:
    def __init__(self, file_path, emb_size=128, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=20):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        self.data = HTNEDataSet(file_path, neg_size, hist_len, directed)

        # the number of the nodes
        self.node_dim = self.data.get_node_dim()

        self.node_emb = Variable(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
            FType), requires_grad=True)

        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

        self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.delta])
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        # the size of the current batch
        batch = s_nodes.size()[0]

        # get the embedding by index
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)

        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times,
                                            h_time_mask)

        # the equation (10) in paper
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

        self.save_node_embeddings('./emb/dblp_htne_attn_%d.emb' % (self.epochs))

    def save_node_embeddings(self, path):
        embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()

if __name__ == '__main__':
    htne = HTNE_a('./dataset/dblp.txt')
    htne.train()
