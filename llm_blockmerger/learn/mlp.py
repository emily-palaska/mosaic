import torch
import torch.nn.init as init

from torch.nn import Module, Linear, Sequential, ReLU
from torch.utils.data import DataLoader, random_split
from time import time

class MLP(Module):
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
            linear = Linear(prev_dim, dim)
            init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            init.zeros_(linear.bias)  # Bias init to zero
            layers.append(linear)
            layers.append(ReLU())
            prev_dim = dim

        self.network = Sequential(*layers)
        self.network.to(self.device)

    def forward(self, x):
        return self.network(x)

def loaders(dataset, batch, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    return train_loader, val_loader

def eval_feed(model, loss_func, loader):
    model.eval()
    with torch.no_grad():
        loss = labels = var = loss_sim = n = 0
        for a, b, c in loader:
            a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)
            ao, bo, co = model(a), model(b), model(c)
            l, lab, v, lsim = loss_func(ao, bo, co)
            loss += l.item(); labels += lab.item(); var += v.item(); loss_sim += lsim.item(); n += 1

    return loss/n, labels/n, var/n, loss_sim/n

def train_feed(model, optimizer, loss_func, loader):
    model.train()
    loss = labels = var = loss_sim = n = 0
    for a, b, c in loader:
        a, b, c = a.to(model.device), b.to(model.device), c.to(model.device)
        optimizer.zero_grad()
        ao, bo, co = model(a), model(b), model(c)
        l, lab, v, lsim = loss_func(ao, bo, co)
        assert not torch.isnan(l).any(), f"Loss is NaN"
        l.backward(); optimizer.step()
        loss += l.item(); labels += lab.item(); var += v.item(); loss_sim += lsim.item(); n += 1
    return loss / n, labels / n, var / n, loss_sim / n

def update_results(loss, labels, var, loss_sim, results, label):
    for key, val in zip(["loss", "labels", "var", "loss_sim"], [loss, labels, var, loss_sim]):
        results[label][key].append(val)

def train(model, train_loader, val_loader, optimizer, loss_func, epochs=10, verbose=True):
    metadata = model.metadata | loss_func.metadata | {"lr": optimizer.param_groups[0]["lr"]}
    results = {
        "metadata": metadata,
        "time": [],
        "train": {"loss": [], "labels": [], "var": [], "loss_sim": []},
        "val":   {"loss": [], "labels": [], "var": [], "loss_sim": []}
    }

    for epoch in range(epochs):
        start = time()
        loss, labels, var, loss_sim = train_feed(model, optimizer, loss_func, train_loader)
        update_results(loss, labels, var, loss_sim, results, "train")
        update_results(*eval_feed(model, loss_func, val_loader), results, "val")
        results["time"].append(time() - start)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}]\t"
                  f"Train Loss: {loss:.3f}\t"
                  f"Sim: {loss_sim:.3f}\t"
                  f"Labels: {labels:.3f}\t"
                  f"Var: {var:.3f}")

    return results