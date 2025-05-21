from integration_database_merge import ready_database_pipeline
from llm_blockmerger.core import load_double_encoded_json, embedding_projection
from torch import tensor
import textwrap, torch

def print_search_results(search_embedding, nearest_neighbors):
    print('')
    print('=' * 60)
    for nearest_neighbor in nearest_neighbors:
        print('-' * 60)
        data = load_double_encoded_json(nearest_neighbor.blockdata)
        neighbor_embedding = embedding_projection(search_embedding, nearest_neighbor.embedding)
        print(nearest_neighbor.id, 'Projection to search norm: ', neighbor_embedding.norm())
        print(textwrap.indent(data['blocks'], '\t'))

    print('=' * 60)

def calculate_new_search_embedding(search_embedding, neighbor_embedding, l=0.35):
    search_projection = embedding_projection(search_embedding, neighbor_embedding)
    remaining_projection = neighbor_embedding - search_projection
    new_search_embedding = neighbor_embedding + l*remaining_projection - search_embedding
    return new_search_embedding / new_search_embedding.norm()

def main():
    embedding_model, vector_db = ready_database_pipeline()
    specification = 'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'
    search_embedding = tensor(embedding_model.encode_strings(specification)[0])

    nearest_neighbors = vector_db.read(search_embedding, limit=3)
    print_search_results(search_embedding, nearest_neighbors)

    search_embedding = calculate_new_search_embedding(search_embedding, nearest_neighbors[0].embedding)

    nearest_neighbors = vector_db.read(search_embedding, limit=3)
    print_search_results(search_embedding, nearest_neighbors)

    search_embedding = calculate_new_search_embedding(search_embedding, nearest_neighbors[0].embedding)

    nearest_neighbors = vector_db.read(search_embedding, limit=3)
    print_search_results(search_embedding, nearest_neighbors)



if __name__ == '__main__':
    main()