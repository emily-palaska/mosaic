# LlmBlockMerger-Diploma
A Master Thesis exploring Automated Code Block Synthesis enhanced by LLMs.


**Dependencies:** Python 3.12<br>
**Contact:** Aimilia Palaska (aimilia.p2@gmail.com)<br>
**License:** Apache 2.0

## ⚡Quickstart
Clone this repository with `git clone https://github.com/emily-palaska/LlmBlockMerger-Diploma`<br>
Install requirements with `pip install -r requirements.txt`<br>

```python
# load
paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
managers = initialize_managers(paths)
llama = LLM(task='question')
embedding_model = LLM(task='embedding')
embeddings = embedding_model.encode_strings(extract_labels(managers, blocks=True))
plot_similarity_matrix(compute_embedding_similarity(embeddings), './plots/similarity_matrix.png')

# store
vector_db = BlockMergerVectorDB(databasetype=HNSWVectorDB, empty=True)
vector_db.create(embeddings=embeddings,
                     blockdata=create_blockdata(managers, embeddings))
# merge
specification = 'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'
print_merge_result(
  specification,
  linear_string_merge(embedding_model=embedding_model, vector_db=vector_db,
                      specification=specification, var_merge=False),
  merge_type='STRING'
)
print_merge_result(
  specification,
  linear_embedding_merge(embedding_model=embedding_model, vector_db=vector_db,
                         specification=specification, var_merge=False),
  merge_type='EMBEDDING'
)
```

## 🧠About
LLM-BlockMerger utilizes LLMs along with a VectrorDB for Retrieval Augemntation to merge Code Blocks from natural language speficiations. It can be divided into 5 functional modules:
- **core**: llm loading/quering, embedding operations and abstract syntax tree analyzers
- **load**: file pre-processing and block extraction based on comments and markdown text
- **store**: VectorDB initialization and functionality for accelerate retrieval of Code Blocks
- **learn**: MLP that applies Transfer Learning to the Embedding Space in order to enforce transitivity relation
- **merge**: two block synthesis mechanisms, based on string alteration or embedding projections

An abstracted high-level flow chart of the mechanism:
<p align=center> <img title="Absttract Flowchart" alt="LLM-BlockMerger" src="plots/system_general_eng.png"> 

## 🧮Embedding Synthesis
This method is a unique element of the mechanism, which leverages vector operations to subtract infromation and iteratively query the VectorDB in order to retrieve blocks that implement a natural language specification. Its interests lies in the adaptation to the LLM's embedding space properties of semantic proximity between vectors.

```python
search_embedding = tensor(embedding_model.encode_strings(specification)[0])
specification_embedding = tensor(embedding_model.encode_strings(specification)[0])
information = specification_embedding.norm().item()

  for _ in range(max_it):
      if information < norm_threshold: break # Break condition: Embedding norm below the norm threshold

      nearest_neighbor = vector_db.read(search_embedding, limit=1)[0]
      if nearest_neighbor is None: break  # Break condition: No neighbors

      neighbor_embedding = nearest_neighbor.embedding
      neighbor_projection = embedding_projection(neighbor_embedding, search_embedding)
      info_projection = embedding_projection(specification_embedding, neighbor_embedding)
      if norm(neighbor_projection) < norm_threshold: break  # Break condition: Perpendicular embeddings

      merge_block_manager.append_doc(nearest_neighbor)

      search_embedding = l * neighbor_projection - search_embedding
      search_embedding /= search_embedding.norm()
      information -= k * info_projection.norm().item()
```


A visual representation of the core idea behind the search embedding rotation:
<p align=center> <img title="Absttract Flowchart" alt="LLM-BlockMerger" src="plots/sphere.png" height=400px> <img title="Absttract Flowchart" alt="LLM-BlockMerger" src="plots/vectors.png" height=400px> 
