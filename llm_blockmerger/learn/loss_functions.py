import torch
from llm_blockmerger.core import pairwise_norm_cos_sim, variance

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
        a = a / a.norm()
        b = b / b.norm()
        c = c / c.norm()
        ab = pairwise_norm_cos_sim(a, b)
        bc = pairwise_norm_cos_sim(b, c)
        ac = pairwise_norm_cos_sim(a, c)
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
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        c = c / c.norm(dim=-1, keepdim=True)

        ab = pairwise_norm_cos_sim(a, b)
        bc = pairwise_norm_cos_sim(b, c)
        ac = pairwise_norm_cos_sim(a, c)

        labels = ab * bc
        threshold = labels.quantile(0.9).item() if self.quantile else self.threshold ** 2
        labels = (labels > threshold).float()
        var_ac = variance(ac)

        # Contrastive loss: pull similar pairs together, push dissimilar apart
        # Positive pair: L = (1 - sim)^2, Negative pair: L = max(0, sim - margin)^2
        positive_loss = labels * (1 - ac) * (1 - ac)
        negative_loss = (1 - labels) * torch.relu(ac - self.margin) * torch.relu(ac - self.margin)
        loss = positive_loss + negative_loss

        if self.mean: loss += (ac.mean() - 0.5) * (ac.mean() - 0.5)
        if self.var: loss -= self.alpha * torch.log(var_ac + 1.0e-12)

        total_loss = loss.mean()
        return total_loss, labels.mean(), var_ac, (positive_loss + negative_loss).mean()