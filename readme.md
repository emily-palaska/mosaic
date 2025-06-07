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
