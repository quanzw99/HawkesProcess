from torch.utils.data import Dataset
import numpy as np
import sys
from bisect import bisect_left
from HawkesDataSet import HawkesDataSet

class NISDataSet(HawkesDataSet):
    def __init__(self, file_path, neg_size, hist_len, directed=False, transform=None):
        super(NISDataSet, self).__init__(file_path, neg_size, hist_len, directed, transform)

        # get s_idx
        self.s_idx_map = {}
        for node in self.node2hist:
            hist = self.node2hist[node]
            node_hist = {}
            for i, item in enumerate(hist):
                key = item
                if key not in node_hist:
                    node_hist[key] = i
            self.s_idx_map[node] = node_hist

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            s_hist = self.node2hist[s_node][0:t_idx]
        else:
            s_hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        s_hist_nodes = [h[0] for h in s_hist]
        s_hist_times = [h[1] for h in s_hist]

        np_s_hist_nodes = np.zeros((self.hist_len,))
        np_s_hist_nodes[:len(s_hist_nodes)] = s_hist_nodes
        np_s_hist_times = np.zeros((self.hist_len,))
        np_s_hist_times[:len(s_hist_times)] = s_hist_times
        np_s_hist_masks = np.zeros((self.hist_len,))
        np_s_hist_masks[:len(s_hist_nodes)] = 1.

        s_idx = self.s_idx_map[t_node].get((s_node, t_time))
        if s_idx - self.hist_len < 0:
            t_hist = self.node2hist[t_node][0:s_idx]
        else:
            t_hist = self.node2hist[t_node][s_idx - self.hist_len:s_idx]

        t_hist_nodes = [h[0] for h in t_hist]
        t_hist_times = [h[1] for h in t_hist]

        np_t_hist_nodes = np.zeros((self.hist_len,))
        np_t_hist_nodes[:len(t_hist_nodes)] = t_hist_nodes
        np_t_hist_times = np.zeros((self.hist_len,))
        np_t_hist_times[:len(t_hist_times)] = t_hist_times
        np_t_hist_masks = np.zeros((self.hist_len,))
        np_t_hist_masks[:len(t_hist_nodes)] = 1.

        neg_nodes = self.negative_sampling().astype(int)
        np_neg_hists_nodes = np.zeros((self.neg_size, self.hist_len))
        np_neg_hists_times = np.zeros((self.neg_size, self.hist_len))
        np_neg_hists_masks = np.zeros((self.neg_size, self.hist_len))
        for i in range(0, self.neg_size):
            neg_node = neg_nodes[i]
            n_idx = self.get_n_idx(self.node2hist[neg_node], t_time)
            if n_idx - self.hist_len < 0:
                n_hist = self.node2hist[neg_node][0:n_idx]
            else:
                n_hist = self.node2hist[neg_node][n_idx - self.hist_len: n_idx]
            n_hist_nodes = [h[0] for h in n_hist]
            n_hist_times = [h[1] for h in n_hist]
            np_neg_hists_nodes[i][:len(n_hist_nodes)] = n_hist_nodes
            np_neg_hists_times[i][:len(n_hist_times)] = n_hist_times
            np_neg_hists_masks[i][:len(n_hist_nodes)] = 1.

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'neg_nodes': neg_nodes,
            's_hist_nodes': np_s_hist_nodes,
            's_hist_times': np_s_hist_times,
            's_hist_masks': np_s_hist_masks,
            't_hist_nodes': np_t_hist_nodes,
            't_hist_times': np_t_hist_times,
            't_hist_masks': np_t_hist_masks,
            'n_hists_nodes': np_neg_hists_nodes,
            'n_hists_times': np_neg_hists_times,
            'n_hists_masks': np_neg_hists_masks,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_n_idx(self, arr, t_time):
        tmp_arr = [item[1] for item in arr]
        n_idx = bisect_left(tmp_arr, t_time)
        return n_idx
