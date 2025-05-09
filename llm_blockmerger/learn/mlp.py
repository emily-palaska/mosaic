import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from llm_blockmerger.learn.dummy_dataset import DummyTripletDataset
from llm_blockmerger.learn.loss_functions import triplet_cross_entropy_loss, transitive_contrastive_loss

class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims=None, loss_function=triplet_cross_entropy_loss):
        assert loss_function in [transitive_contrastive_loss, triplet_cross_entropy_loss], \
            f'{loss_function} is not a valid loss function'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims else [64, 32]
        self.loss_function = loss_function

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

def train(model, train_loader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss, num_batches = 0.0, 0
        total_var = 0.0

        for batch in train_loader:
            a, b, c = batch
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)

            optimizer.zero_grad()
            a_out, c_out = model(a), model(c)
            loss = model.loss_function(a, b, c, a_out, c_out)
            assert not torch.isnan(loss).any(), f"Loss is NaN at epoch {epoch + 1}"
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_var += torch.var(c_out).item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_var = total_var / num_batches
        print(f'\tEpoch [{epoch + 1}/{epochs}]\t Loss: {avg_loss:.4f}\t Variance: {avg_var:.4f}')

def main():
    feat_dim, num_samples, batch_size, lr, epochs, layer_dims = 128, 1000, 32, 0.001, 10, [64, 32]

    model = MLP(input_dim=feat_dim, layer_dims=layer_dims, loss_function=triplet_cross_entropy_loss)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = DummyTripletDataset(num_samples=num_samples, feat_dim=feat_dim)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train(model, train_loader, optimizer, epochs=epochs)

if __name__ == "__main__":
    main()
