from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

class EmbeddingModel:
    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model)
        self.labels = None
        self.embeddings = None
        self.similarity_matrix = None
        self.embedding_dim = None

    def encode(self, labels):
        """
        Encodes a list of sentences and stores the embeddings.
        :param labels: numpy array of labels to be encoded
        """
        self.labels = labels
        self.embeddings = self.model.encode(labels)
        self.embedding_dim = self.embeddings.shape[1] if self.embeddings.ndim == 2 else self.embeddings.shape[1:]
        return self.embeddings

    def plot_sim(self):
        # Plot the similarity matrix
        plt.figure(figsize=(12, 12))
        plt.imshow(self.similarity_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Similarity')

        # Annotate matrix
        plt.title("Similarity Matrix")
        for i in range(self.similarity_matrix.shape[0]):
            for j in range(self.similarity_matrix.shape[1]):
                plt.text(j, i, f"{self.similarity_matrix[i, j]:.2f}",
                         ha="center", va="center", color="black")

        # Show the plot
        plt.tight_layout()
        plt.savefig('../plots/similarity_matrix.png')

    def similarity(self, plot=True):
        """
        Computes a similarity matrix for all pairs of embeddings.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings are not initialized. Call 'encode' first.")
        self.similarity_matrix = util.pytorch_cos_sim(self.embeddings, self.embeddings)

        if plot: self.plot_sim()
