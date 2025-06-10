import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from llm_blockmerger.learn.dummy_dataset import DummyTripletDataset
from llm_blockmerger.learn.loss_functions import TransitiveCrossEntropyLoss, transitive_contrastive_loss, vector_variance, pairwise_norm_cos_sim
from llm_blockmerger.learn.visualization import visualize_results

class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims else [64, 32]
        self.metadata = {
            "input_dim": self.input_dim,
            "layer_dims": str(self.layer_dims),
        }

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


def train(model, train_loader, optimizer, loss_function, epochs=10, verbose=True):
    model.train()
    metadata = model.metadata | loss_function.metadata | {"lr": optimizer.param_groups[0]["lr"]}
    results = {"loss": [], "labels": [], "var": [], "loss_sim": [], "time":[], "metadata": metadata}
    for epoch in range(epochs):
        start_time = time.time()
        loss = labels =  var = loss_sim = num_batches = 0

        for batch in train_loader:
            a, b, c = batch
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)

            optimizer.zero_grad()
            a_out, b_out, c_out = model(a), model(b), model(c)

            batch_loss, batch_labels, batch_var, batch_loss_sim = loss_function(a_out, b_out, c_out)
            assert not torch.isnan(batch_loss).any(), f"Loss is NaN at epoch {epoch + 1}"
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            labels += batch_labels.item()
            var += batch_var.item()
            loss_sim += batch_loss_sim.item()
            num_batches += 1
        results["loss"].append(loss / num_batches)
        results["labels"].append(labels / num_batches)
        results["var"].append(var / num_batches)
        results["loss_sim"].append(loss_sim / num_batches)
        results["time"].append(time.time() - start_time)
        if verbose:
            print(f'Epoch [{epoch + 1}/{epochs}]', end='\t')
            print(f'Loss: {results["loss"][-1]: .3f}', end='\t')
            print(f'Sim Loss: {results["loss_sim"][-1]: .3f}', end='\t')
            print(f'Labels: {results["labels"][-1]: .3f}', end='\t')
            print(f'Variance: {results["var"][-1]: .3f}', end='\n')

    return results

def main():
    feat_dim, num_samples, batch_size, lr, epochs, layer_dims = 128, 1000, 1000, 0.001, 10, [64, 32]

    model = MLP(input_dim=feat_dim, layer_dims=layer_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = DummyTripletDataset(num_samples=num_samples, feat_dim=feat_dim)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    results = train(model, train_loader, optimizer, epochs=epochs, loss_function=TransitiveCrossEntropyLoss())
    visualize_results(results)

if __name__ == "__main__":
    main()
