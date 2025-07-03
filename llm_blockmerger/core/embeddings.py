import matplotlib.pyplot as plt
import torch
from torch import Tensor, rand, norm


def projection(a, b):
    if not isinstance(a, torch.Tensor): a = torch.tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.tensor(b)

    if torch.all(a == 0): return None
    dot = torch.dot(b, a) / torch.dot(a, a)
    return dot * a

def norm_cos_sim(batch1, batch2=None, dtype=torch.float):
    if batch2 is None: batch2 = batch1

    if not isinstance(batch1, torch.Tensor): batch1 = torch.tensor(batch1, dtype=dtype)
    if not isinstance(batch2, torch.Tensor): batch2 = torch.tensor(batch2, dtype=dtype)

    a = batch1.unsqueeze(1)
    b = batch2.unsqueeze(0)

    dot = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    denom = norm_a * norm_b + 1.0e-8

    cos_sim = dot / denom
    return (cos_sim + 1) / 2


def variance(batch):
    assert batch.shape[0] != 1, "Expected batch of vectors, not singular vector"
    mean = torch.mean(batch, dim=0)
    squared_diffs = (batch - mean) ** 2
    var_per_dim = torch.mean(squared_diffs, dim=0)
    return torch.mean(var_per_dim)


def plot_sim(sim_mat, path='../plots/similarity_matrix.png'):
    plt.figure()
    plt.imshow(sim_mat, cmap='cividis', interpolation='nearest')
    plt.colorbar(label='Ομοιότητα')
    plt.axis('off')

    # Annotate the matrix with similarity values - ONLY for small dimensions due to visibility
    if sim_mat.shape[0] <= 20:
        for i in range(sim_mat.shape[0]):
            for j in range(sim_mat.shape[1]):
                plt.text(j, i, f"{sim_mat[i, j]:.2f}",
                         ha="center", va="center", color="black")

    #plt.title("Similarity Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def norm_batch(batch: torch.Tensor):
    return batch / batch.norm(dim=-1, keepdim=True)


def pivot_rotation(q: Tensor, n: Tensor, s: Tensor, i: Tensor, k: float, l: float, method:str|None=None):
    n_proj = projection(n, s)
    i_proj = projection(q, n)

    if method == 'rnd': s = rand(s.shape)
    elif method == 'rev': s = - n
    else: s = l * n_proj - s
    s /= s.norm()
    i -= k * i_proj.norm().item()
    return s, i, norm(n_proj).item()
