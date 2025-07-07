from numpy import mean, std

from mosaic.core import LLM, norm_cos_sim
from tests.core.utils import remove_words
from tests.data.empirical import queries, programs, program_blocks
from tests.data.stopwords import stopwords

def empirical(verbose=True):
    program_lines = [len(program.splitlines()) for program in programs]
    q_words = [len(q.split()) for q in remove_words(queries, stopwords)]
    assert len(program_blocks) == len(q_words)

    block_dist = [pb / qw for qw, pb in zip(q_words, program_blocks)]
    if verbose: print(f"Empirical Blocks per Query word: {mean(block_dist)} +- {std(block_dist)}")
    lines_dist = [pl / qw for qw, pl in zip(q_words, program_lines)]
    if verbose: print(f"Empirical Lines per Query word: {mean(lines_dist)} +- {std(lines_dist)}")

    model = LLM(task="embedding")
    qs = model.encode(queries)
    ps = model.encode(programs)
    sims = [norm_cos_sim(q, p) for q, p in zip(qs, ps)]
    if verbose: print(f"Empirical Similarity: {mean(sims)} +- {std(sims)}")
    return block_dist, lines_dist

if __name__ == "__main__":
    empirical()