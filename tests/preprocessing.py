import os
os.chdir("../")

from llm_blockmerger.load import init_managers, flatten_labels, create_blockdata, nb_variables
from llm_blockmerger.store import BlockDB
from llm_blockmerger.core import plot_sim, pairwise_norm_cos_sim, LLM

def main():
    paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    #paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    managers = init_managers(paths)

    llama = LLM(task='question')
    for i, manager in enumerate(managers):
        nb_variables(manager, llama)
        print(manager)

    model = LLM(task='embedding')
    embeddings = model.encode(flatten_labels(managers, code=True))
    plot_sim(pairwise_norm_cos_sim(embeddings), './plots/similarity_matrix.png')

    db = BlockDB(empty=True)
    db.create(embeddings=embeddings, blockdata=create_blockdata(managers, embeddings))
    assert db.num_docs() == len(embeddings), f'{db.num_docs()} != {len(embeddings)}'


if __name__ == '__main__':
    main()