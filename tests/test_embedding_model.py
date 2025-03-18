import pytest, sys
sys.path.append("..")
from llm_blockmerger.store.embedding_similarity import *

@pytest.fixture
def labels():
    return ['label 1', 'label 2', 'label 3', 'label 4', 'label 5']

def test_embedding_model_default(labels):
    embedding_model = initialize_model()
    assert isinstance(embedding_model, SentenceTransformer)
    embeddings = encode_labels(embedding_model, labels)
    for embedding in embeddings:
        assert embedding.shape == (384,)