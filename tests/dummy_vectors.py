from re import search

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
        _, angles = to_polar_coordinates(nearest_neighbor.embedding)
        print('Angle sample: ', angles[:5])
        print(textwrap.indent(data['blocks'], '\t'))

    print('=' * 60)

def to_polar_coordinates(x: torch.Tensor):
    n = x.shape[0]
    r = torch.norm(x)

    angles = []
    for i in range(n - 1):
        denom = torch.norm(x[i:])
        if i == n - 2:
            angle = torch.atan2(x[-1], x[-2])
        else:
            angle = torch.acos(x[i] / denom)
        angles.append(angle)

    angles = torch.stack(angles)
    return r, angles


def to_cartesian_coordinates(r: torch.Tensor, angles: torch.Tensor):
    n = angles.shape[0] + 1
    x = torch.zeros(n)

    prod_sin = r
    for i in range(n):
        if i == 0:
            x[i] = r * torch.cos(angles[0])
        elif i < n - 1:
            prod_sin *= torch.sin(angles[i - 1])
            x[i] = prod_sin * torch.cos(angles[i])
        else:
            prod_sin *= torch.sin(angles[-1])
            x[i] = prod_sin
    return x

def calculate_new_search_embedding(search_embedding, neighbor_embedding, l=0.35):
    search_norm, search_angles = to_polar_coordinates(search_embedding)
    neighbor_norm, neighbor_angles = to_polar_coordinates(neighbor_embedding)

    #projection_angles = embedding_projection(neighbor_angles, search_angles)
    #new_angles = 2*projection_angles - 0.8*search_angles -0.2*neighbor_angles
    #new_search_embedding=to_cartesian_coordinates(search_norm, new_angles)

    projection = embedding_projection(neighbor_embedding, search_embedding)
    a = 1.4
    new_search_embedding = a*projection - search_embedding

    # convergence search_embedding*remaining_projection
    #new_search_embedding = 0.5*search_embedding + 0.5*neighbor_embedding

    print('search_embedding', search_embedding[:5])
    print('neighbor_embedding', neighbor_embedding[:5])
    print('search_angles', search_angles[:5])
    print('neighbor_angles', neighbor_angles[:5])
    #print('projection_angles', projection_angles[:5])
    #print('new_angles', new_angles[:5])
    print('new_search_embedding', new_search_embedding[:5])
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