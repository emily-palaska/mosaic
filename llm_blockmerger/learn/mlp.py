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

def normalized_cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
    return (cos_sim + 1) / 2

def transitive_contrastive_loss(a, b, c, threshold=0.8, margin=1.0):
    # Normalized cosine distances
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ca = normalized_cosine_similarity(c, a)

    label = (torch.sqrt(ab * bc) > threshold).float()
    loss_similar = (1 - label) * torch.pow(ca, 2)
    loss_dissimilar = label * torch.pow(torch.clamp(margin - ca, min=0.0), 2)

    #print(f'\t Label distribution: {torch.sum(label, dim=0) / label.shape[0]}')
    #print(f'\t Similar: {torch.mean(loss_similar)} Dissimilar: {torch.mean(loss_dissimilar)}')
    return torch.mean(0.5 * (loss_similar + loss_dissimilar))

def train(model, train_loader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            a, b, c = batch
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)

            optimizer.zero_grad()
            emb_a, emb_b, emb_c = model(a), model(b), model(c)
            loss = transitive_contrastive_loss(emb_a, emb_b, emb_c,)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{epochs}]\t Loss: {avg_loss:.4f}')

def main():
    feat_dim = 128
    model = MLP(input_dim=feat_dim, layer_dims=[64, 32])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = DummyTripletDataset(num_samples=32, feat_dim=feat_dim)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    train(model, train_loader, optimizer, epochs=10)

if __name__ == "__main__":
    main()
