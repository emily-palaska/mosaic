from llm_blockmerger.store import BlockDB
from llm_blockmerger.core import LLM, print_synthesis
from llm_blockmerger.merge import string_synthesis, embedding_synthesis


def main():
    spec = 'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'
    model = LLM(task='embedding')
    db = BlockDB(empty=False)
    synthesis = string_synthesis(model, db, spec)
    print_synthesis(spec, synthesis, title='STRING')
    synthesis = embedding_synthesis(model, db, spec)
    print_synthesis(spec, synthesis, title='EMBEDDING')


if __name__ == '__main__':
    main()