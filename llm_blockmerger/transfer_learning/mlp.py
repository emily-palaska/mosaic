import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dummy_dataset import DummyTripletDataset

class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims else [64, 32]
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.network = nn.Sequential(*layers)
        self.network.to(self.device)

    def forward(self, x):
        return self.network(x)

def transitive_contrastive_loss(a, b, c, label_ab, label_bc, margin=0.2, alpha=0.5):
    # Compute cosine distances
    cos_dist_ab = 1 - F.cosine_similarity(a, b, dim=1)
    cos_dist_bc = 1 - F.cosine_similarity(b, c, dim=1)
    cos_dist_ac = 1 - F.cosine_similarity(a, c, dim=1)

    # Pairwise losses
    loss_ab = (label_ab * torch.pow(cos_dist_ab, 2) +
               (1 - label_ab) * torch.pow(torch.clamp(margin - cos_dist_ab, min=0.0), 2))
    loss_bc = (label_bc * torch.pow(cos_dist_bc, 2) +
               (1 - label_bc) * torch.pow(torch.clamp(margin - cos_dist_bc, min=0.0), 2))

    # Transitivity term
    trans_loss = (label_ab * label_bc) * torch.pow(cos_dist_ac, 2)

    return (1 - alpha) * (loss_ab.mean() + loss_bc.mean()) + alpha * trans_loss.mean()

def train(model, train_loader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            a, b, c, label_ab, label_bc = batch
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)
            label_ab, label_bc = label_ab.to(model.device), label_bc.to(model.device)

            optimizer.zero_grad()
            emb_a, emb_b, emb_c = model(a), model(b), model(c)

            loss = transitive_contrastive_loss(
                emb_a, emb_b, emb_c,
                label_ab, label_bc,
                margin=0.2, alpha=0.5
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{epochs}]\t Loss: {avg_loss:.4f}')

def main():
    model = MLP(input_dim=128, layer_dims=[64, 32])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = DummyTripletDataset(num_samples=1000, feat_dim=128)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    train(model, train_loader, optimizer, epochs=10)

if __name__ == "__main__":
    main()
