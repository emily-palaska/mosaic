import torch
from torch.utils.data import Dataset

class DummyTripletDataset(Dataset):
    def __init__(self, num_samples=1000, feat_dim=128, std=5.0):
        self.num_samples = num_samples
        self.feat_dim = feat_dim
        self.data = torch.randn(num_samples * 3, feat_dim) * std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a = self.data[3*idx]
        b = self.data[3*idx + 1]
        c = self.data[3*idx + 2]
        return a, b, c

