import torch
from llm_blockmerger.core import pairwise_norm_cos_sim, vector_variance

def transitive_contrastive_loss(a, b, c, a_out, c_out, threshold=0.98, alpha=1000):
    ab = pairwise_norm_cos_sim(a, b)
    bc = pairwise_norm_cos_sim(b, c)
    ac_out = pairwise_norm_cos_sim(a_out, c_out)

    labels = (ab * bc > threshold ** 2).float()

    similar = (1 - labels) * ac_out
    dissimilar = labels * (1 - ac_out)

    loss = alpha * torch.var(c_out) - similar - dissimilar
    return loss.mean()

def transitive_cross_entropy_loss(a, b, c, threshold=0.5, alpha=0.1):
    ab = pairwise_norm_cos_sim(a, b)
    bc = pairwise_norm_cos_sim(b, c)
    ac = pairwise_norm_cos_sim(a, c)
    assert torch.min(ac) >=0 and torch.max(ac) <= 1

    labels = ab * bc
    threshold = labels.quantile(0.9).item()
    labels = (labels > threshold).float()

    var_ac = vector_variance(ac)
    similar = labels * torch.log(ac + 1.0e-12)
    dissimilar = (1 - labels) * torch.log(1 - ac + 1.0e-12)

    #loss = - alpha * torch.log(var_ac + 1.0e-12) - similar - dissimilar
    loss = (ac.mean() - 0.5) * (ac.mean() - 0.5) - alpha * torch.log(var_ac + 1.0e-12) - similar - dissimilar
    loss_similar = - similar - dissimilar
    return loss.mean(), labels.mean().item(), var_ac.item(), loss_similar.mean().item()
