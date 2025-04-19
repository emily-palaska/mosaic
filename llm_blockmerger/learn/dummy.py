import torch

a = torch.tensor([0.6, 0.3])
print((a > 0.5 + 0.05).float())