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

def eval_feed(model, loss_function, loader):
    model.eval()
    with torch.no_grad():
        loss = labels = var = loss_sim = n = 0
        for a, b, c in loader:
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)
            ao, bo, co = model(a), model(b), model(c)
            l, lab, v, lsim = loss_function(ao, bo, co)
            loss += l.item(); labels += lab.item(); var += v.item(); loss_sim += lsim.item(); n += 1

    return loss/n, labels/n, var/n, loss_sim/n

def train_feed(model, optimizer, loss_function, loader):
    model.train()
    loss = labels = var = loss_sim = n = 0
    for a, b, c in loader:
        a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)
        optimizer.zero_grad()
        ao, bo, co = model(a), model(b), model(c)
        l, lab, v, lsim = loss_function(ao, bo, co)
        assert not torch.isnan(l).any(), f"Loss is NaN"
        l.backward(); optimizer.step()
        loss += l.item(); labels += lab.item(); var += v.item(); loss_sim += lsim.item(); n += 1
    return loss / n, labels / n, var / n, loss_sim / n

def train(model, train_loader, val_loader, optimizer, loss_function, epochs=10, verbose=True):
    model.train()
    metadata = model.metadata | loss_function.metadata | {"lr": optimizer.param_groups[0]["lr"]}
    results = {
        "metadata": metadata,
        "time": [],
        "train": {"loss": [], "labels": [], "var": [], "loss_sim": []},
        "val":   {"loss": [], "labels": [], "var": [], "loss_sim": []}
    }

    for epoch in range(epochs):
        start = time.time()

        loss, labels, var, loss_sim = train_feed(model, optimizer, loss_function, train_loader)
        for key, val in zip(["loss", "labels", "var", "loss_sim"], [loss, labels, var, loss_sim]):
            results["train"][key].append(val)

        val_loss, val_labels, val_var, val_sim = eval_feed(model, loss_function, val_loader)
        for key, val in zip(["loss", "labels", "var", "loss_sim"], [val_loss, val_labels, val_var, val_sim]):
            results["val"][key].append(val)

        results["time"].append(time.time() - start)
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}]\t"
                  f"Train Loss: {loss:.3f}\t"
                  f"Sim: {loss_sim:.3f}\t"
                  f"Labels: {labels:.3f}\t"
                  f"Var: {var:.3f}")

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
