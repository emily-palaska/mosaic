# MOSAIC: Model-based Orchestration of Semantic Assembly through Informed Composition
Developed as part of Master Thesis titled "Traceable linear code synthesis at scale"

**Dependencies:** Python 3.12<br>
**Contact:** Aimilia Palaska (aimilia.p2@gmail.com)<br>
**License:** Apache 2.0

## ⚡Quickstart
Clone this repository with `git clone https://github.com/emily-palaska/LlmBlockMerger-Diploma`<br>
Install requirements with `pip install -r requirements.txt`<br>

### Example of pre-processing pipeline

```python
from llm_blockmerger.load import init_managers, flatten_labels, create_blockdata, nb_variables
from llm_blockmerger.store import BlockDB
from llm_blockmerger.core import plot_sim, norm_cos_sim, LLM

paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
managers = init_managers(paths)

llama = LLM(task='question')
for i, manager in enumerate(managers):
    nb_variables(manager, llama)
    print(manager)

model = LLM(task='embedding')
embeddings = model.encode(flatten_labels(managers, code=True))
plot_sim(norm_cos_sim(embeddings), './plots/similarity_matrix.png')

db = BlockDB(empty=True)
db.create(embeddings=embeddings, blockdata=create_blockdata(managers, embeddings))
```
### Example of merging scenario

```python
from llm_blockmerger.store import BlockDB
from llm_blockmerger.core import LLM, print_synthesis
from llm_blockmerger.merge import string_synthesis, embedding_synthesis

spec = 'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'
model = LLM(task='embedding')
db = BlockDB(empty=False)
synthesis = string_synthesis(model, db, spec)
print_synthesis(spec, synthesis, title='STRING')
synthesis = embedding_synthesis(model, db, spec)
print_synthesis(spec, synthesis, title='EMBEDDING')
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
from llm_blockmerger.load import BlockManager
from llm_blockmerger.core import projection, LLM, ast_io_split
from llm_blockmerger.merge.merger import merge_variables
from llm_blockmerger.merge.order import  io_order
from llm_blockmerger.store import BlockDB
from torch import norm, tensor

def embedding_synthesis(model: LLM, db: BlockDB, spec: str, k=0.9, l=1.4, max_it=10, t=0.05, var=True):
    synthesis = BlockManager()
    s = tensor(model.encode(spec)[0])
    spec_emb = tensor(model.encode(spec)[0])
    i = spec_emb.norm().item()

    for _ in range(max_it):
        print(f'Search embedding: {s.norm().item(): .2f}, Information: {i: .2f}')
        if i < t: break # Break condition: Embedding norm below the norm threshold

        nn = db.read(s, limit=1)[0]
        if nn is None: break  # Break condition: No neighbors

        n = nn.embedding
        n_proj = projection(n, s)
        i_proj = projection(spec_emb, n)
        if norm(n_proj) < t: break  # Break condition: Perpendicular embeddings

        synthesis.append_doc(nn)

        s = l * n_proj - s
        s /= s.norm()
        i -= k * i_proj.norm().item()

    synthesis.rearrange(io_order(ast_io_split(synthesis)))
    return merge_variables(model, synthesis) if var else synthesis
```


A visual representation of the core idea behind the search embedding rotation:
<p align=center> <img title="Absttract Flowchart" alt="LLM-BlockMerger" src="plots/sphere.png" height=350px> <img title="Absttract Flowchart" alt="LLM-BlockMerger" src="plots/vectors.png" height=350px> 
