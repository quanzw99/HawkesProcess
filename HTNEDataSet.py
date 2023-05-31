from torch.utils.data import Dataset
import numpy as np
import sys
from HawkesDataSet import HawkesDataSet

class HTNEDataSet(HawkesDataSet):
    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        neg_nodes = self.negative_sampling()

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_masks': np_h_masks,
            'neg_nodes': neg_nodes,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample