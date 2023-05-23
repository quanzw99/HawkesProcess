import torch
import HTNEDataSet

FType = torch.FloatTensor
LType = torch.LongTensor
torch.set_num_threads(1)

class HTNE_a:
    def __init__(self, file_path, emb_size=128, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=50):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        self.data = HTNEDataSet(file_path, neg_size, hist_len, directed)


if __name__ == '__main__':
    htne = HTNE_a('./dataset/dblp.txt')
