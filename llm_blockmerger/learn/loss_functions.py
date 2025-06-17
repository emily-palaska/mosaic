import torch
from llm_blockmerger.core import norm_cos_sim, variance, norm_batch
from torch.nn.functional import leaky_relu

class TransitiveCrossEntropy:
    def __init__(self, threshold=0.9, alpha=0.1, quantile=True, mean=True, var=True):
        self.threshold = threshold
        self.alpha = alpha
        self.mean = mean
        self.var = var
        self.quantile = quantile
        self.metadata = {
            "loss_function": "Transitive CE",
            "threshold": self.threshold,
            "alpha": self.alpha,
            "mean": self.mean,
            "var": self.var,
            "quantile": self.quantile,
        }

    def __call__(self, a, b, c):
        a, b, c = norm_batch(a), norm_batch(b), norm_batch(c)
        ab, bc, ac = norm_cos_sim(a, b), norm_cos_sim(b, c), norm_cos_sim(c, a)

        assert torch.min(ac) >= 0 and torch.max(ac) <= 1

        labels = ab * bc
        threshold = labels.quantile(0.9).item() if self.quantile else self.threshold ** 2
        labels = (labels > threshold).float()
        var_ac = variance(ac)

        similar = labels * torch.log(ac + 1.0e-12)
        dissimilar = (1 - labels) * torch.log(1 - ac + 1.0e-12)
        loss = - similar - dissimilar

        if self.mean: loss += (ac.mean() - 0.5) * (ac.mean() - 0.5)
        if self.var: loss -= self.alpha * torch.log(var_ac + 1.0e-12)

        loss_similar = - similar - dissimilar
        return loss.mean(), labels.mean(), var_ac, loss_similar.mean()


class TransitiveContrastive:
    def __init__(self, threshold=0.9, alpha=0.1, quantile=True, margin=0.2, mean=True, var=True):
        self.threshold = threshold
        self.alpha = alpha
        self.margin = margin
        self.mean = mean
        self.var = var
        self.quantile = quantile
        self.metadata = {
            "loss_function": "Transitive Contrastive",
            "threshold": self.threshold,
            "alpha": self.alpha,
            "margin": self.margin,
            "mean": self.mean,
            "var": self.var,
            "quantile": self.quantile,
        }

    def __call__(self, a, b, c):
        a, b, c = norm_batch(a), norm_batch(b), norm_batch(c)
        ab, bc, ac = norm_cos_sim(a, b), norm_cos_sim(b, c), norm_cos_sim(c, a)

        labels = ab * bc
        threshold = labels.quantile(0.9).item() if self.quantile else self.threshold ** 2
        labels = (labels > threshold).float()
        var_ac = variance(ac)

        # Contrastive loss: pull similar pairs together, push dissimilar apart
        # Positive pair: L = (1 - sim)^2, Negative pair: L = max(0, sim - margin)^2
        positive_loss = labels * (1 - ac) * (1 - ac)
        negative_loss = (1 - labels) * leaky_relu(ac - self.margin) * leaky_relu(ac - self.margin)

        if torch.isnan(negative_loss).any():
            nan_mask = torch.isnan(negative_loss)
            indices = nan_mask.nonzero(as_tuple=True)
            print("NaN found at positions:", indices)
            print("ac at NaN positions:", ac[indices])
            for i, j in zip(indices[0], indices[1]):
                print(f"\na[{i}]: {a[i]}")
                print(f"c[{j}]: {c[j]}")
            exit(1)

        loss = positive_loss + negative_loss

        if self.mean: loss += (ac.mean() - 0.5) * (ac.mean() - 0.5)
        if self.var: loss -= self.alpha * torch.log(var_ac + 1.0e-12)

        total_loss = loss.mean()
        return total_loss, labels.mean(), var_ac, (positive_loss + negative_loss).mean()

