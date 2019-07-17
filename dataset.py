import numpy as np
import pickle
import torch

from torch.utils.data import Dataset 


class EarthquakeDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        with open(src_file, 'rb') as f:
            self.signals = pickle.load(f)     

        with open(tgt_file, 'rb') as f:
            self.targets = pickle.load(f)

    def __getitem__(self, index):
        signals = torch.FloatTensor(self.signals[index])
        target = torch.FloatTensor([self.targets[index]])
        print ('signals size',signals.size())
        print ('targets size', target.size())
        return {
            'id': index,
            'signals': signals,
            'target': target,
        } 

    def __len__(self):
        return len(self.signals)

