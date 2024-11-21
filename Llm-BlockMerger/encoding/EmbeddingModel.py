from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

class EmbeddingModel:
    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.similarity_matrix = None

    def encode(self, labels):
        """
        Encodes a list of sentences and stores the embeddings.
        :param labels: numpy array of labels to be encoded
        """
        self.embeddings = self.model.encode(labels)

    def similarity(self, plot=True):
        """
        Computes a similarity matrix for all pairs of embeddings.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings are not initialized. Call 'encode' first.")
        self.similarity_matrix = util.pytorch_cos_sim(self.embeddings, self.embeddings)

        if plot:
            # Plot the similarity matrix
            plt.figure(figsize=(12, 12))
            plt.imshow(self.similarity_matrix, cmap='coolwarm', interpolation='nearest')
            plt.colorbar(label='Similarity')

            # Add labels (optional)
            plt.title("Similarity Matrix")

            # Annotate matrix (optional)
            for i in range(self.similarity_matrix.shape[0]):
                for j in range(self.similarity_matrix.shape[1]):
                    plt.text(j, i, f"{self.similarity_matrix[i, j]:.2f}",
                             ha="center", va="center", color="black")

            # Show the plot
            plt.tight_layout()
            plt.savefig('./plots/similarity_matrix.png')