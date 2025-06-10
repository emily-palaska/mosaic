import torch
from llm_blockmerger.core import pairwise_norm_cos_sim, vector_variance

class TransitiveCrossEntropyLoss:
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
        ab = pairwise_norm_cos_sim(a, b)
        bc = pairwise_norm_cos_sim(b, c)
        ac = pairwise_norm_cos_sim(a, c)
        assert torch.min(ac) >= 0 and torch.max(ac) <= 1

        labels = ab * bc
        threshold = labels.quantile(0.9).item() if self.quantile else self.threshold ** 2
        labels = (labels > threshold).float()
        var_ac = vector_variance(ac)

        similar = labels * torch.log(ac + 1.0e-12)
        dissimilar = (1 - labels) * torch.log(1 - ac + 1.0e-12)
        loss = - similar - dissimilar

        if self.mean: loss += (ac.mean() - 0.5) * (ac.mean() - 0.5)
        if self.var: loss -= self.alpha * torch.log(var_ac + 1.0e-12)

        loss_similar = - similar - dissimilar
        return loss.mean(), labels.mean(), var_ac, loss_similar.mean()


def transitive_contrastive_loss(a, b, c, a_out, c_out, threshold=0.98, alpha=1000):
    ab = pairwise_norm_cos_sim(a, b)
    bc = pairwise_norm_cos_sim(b, c)
    ac_out = pairwise_norm_cos_sim(a_out, c_out)

    labels = (ab * bc > threshold ** 2).float()

    similar = (1 - labels) * ac_out
    dissimilar = labels * (1 - ac_out)

    loss = alpha * torch.var(c_out) - similar - dissimilar
    return loss.mean()
