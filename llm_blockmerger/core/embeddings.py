from sentence_transformers import util
import matplotlib.pyplot as plt
import torch

def embedding_projection(current_embedding, neighbor_embedding):
    if not isinstance(current_embedding, torch.Tensor): current_embedding = torch.tensor(current_embedding)
    if not isinstance(neighbor_embedding, torch.Tensor): neighbor_embedding = torch.tensor(neighbor_embedding)

    if torch.all(current_embedding == 0): return None
    inner_product = torch.dot(neighbor_embedding, current_embedding) / torch.dot(current_embedding, current_embedding)
    return inner_product * current_embedding

def compute_embedding_similarity(embeddings):
    return util.pytorch_cos_sim(embeddings, embeddings)

def plot_similarity_matrix(similarity_matrix, save_path='../plots/similarity_matrix.png'):
    plt.figure(figsize=(12, 12))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Similarity')

    # Annotate the matrix with similarity values - ONLY for small dimensions due to visibility
    if similarity_matrix.shape[0] <= 20:
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                         ha="center", va="center", color="black")

    plt.title("Similarity Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()