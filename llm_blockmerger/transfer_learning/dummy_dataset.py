import torch
from torch.utils.data import Dataset

class DummyTripletDataset(Dataset):
    def __init__(self, num_samples=1000, feat_dim=128):
        self.num_samples = num_samples
        self.feat_dim = feat_dim
        # Generate random features
        self.data = torch.randn(num_samples * 3, feat_dim)  # Enough for all triplets
        # Random binary labels (40% similar pairs)
        self.labels_ab = torch.randint(0, 2, (num_samples,), dtype=torch.float) * 0.6
        self.labels_bc = torch.randint(0, 2, (num_samples,), dtype=torch.float) * 0.6

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a = self.data[3*idx]
        b = self.data[3*idx + 1]
        c = self.data[3*idx + 2]
        return a, b, c, self.labels_ab[idx], self.labels_bc[idx]

