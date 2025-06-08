import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from llm_blockmerger.learn.dummy_dataset import DummyTripletDataset
from llm_blockmerger.learn.loss_functions import transitive_cross_entropy_loss, transitive_contrastive_loss, vector_variance, pairwise_norm_cos_sim


class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims else [64, 32]

        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in self.layer_dims:
            linear = nn.Linear(prev_dim, dim)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)  # Bias init to zero
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = dim

        self.network = nn.Sequential(*layers)
        self.network.to(self.device)

    def forward(self, x):
        return self.network(x)


def train(model, train_loader, optimizer, loss_function, epochs=10):
    model.train()

    all_losses, all_variances = [], []
    for epoch in range(epochs):
        total_loss, num_batches = 0.0, 0
        labels, var = None, None


        for batch in train_loader:
            a, b, c = batch
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)

            optimizer.zero_grad()
            a_out, b_out, c_out = model(a), model(b), model(c)

            loss, labels, var, loss_sim = loss_function(a_out, b_out, c_out)
            assert not torch.isnan(loss).any(), f"Loss is NaN at epoch {epoch + 1}"
            loss.backward()
            optimizer.step()

            total_loss += loss_sim

            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{epochs}]\t Loss: {avg_loss:.4f} Labels: {labels: .4f}\t Variance (c): {var: .4f}')
        all_losses.append(avg_loss)
        all_variances.append(var)
    plt.figure()
    plt.plot(all_losses)
    plt.title('Losses')
    plt.show()
    plt.figure()
    plt.plot(all_variances)
    plt.title('Variances')
    plt.show()


def main():
    feat_dim, num_samples, batch_size, lr, epochs, layer_dims = 128, 1000, 1000, 0.001, 10, [64, 32]

    model = MLP(input_dim=feat_dim, layer_dims=layer_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = DummyTripletDataset(num_samples=num_samples, feat_dim=feat_dim)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train(model, train_loader, optimizer, epochs=epochs)

if __name__ == "__main__":
    main()
