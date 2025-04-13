from sentence_transformers import util
import matplotlib.pyplot as plt
import numpy as np

def embedding_projection(current_embedding, neighbor_embedding):
    if np.all(current_embedding == 0):
        return None
    inner_product = np.dot(neighbor_embedding, current_embedding) / np.dot(current_embedding, current_embedding)
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