from numpy import mean, std

from scipy.stats import wasserstein_distance
from tests.data import merging_quan, stopwords, queries
from tests.core.utils import remove_words
from tests.calculations.empirical import empirical

def key_dist(key:str, q_words:list):
    counters = {method: m_dict[key] for method, m_dict in merging_quan.items()}
    return {m: [c / qw for c, qw in zip(counter, q_words)] for m, counter in counters.items()}

def merging_distributions():
    q_words = [len(q.split()) for q in remove_words(queries, stopwords)]
    block_dist = key_dist('blocks', q_words)
    lines_dist = key_dist('lines', q_words)
    block_dist["emp"], lines_dist["emp"] = empirical(verbose=False)

    print("VALUE RANGES")
    print("Method\t\tBlock Num / Query Words")
    for m, dist in block_dist.items(): print(f'\t{m}\t\t{mean(dist):.3f} ± {std(dist):.3f}')

    print("Method\t\tLine Num / Query Words")
    for m, dist in lines_dist.items(): print(f'\t{m}\t\t{mean(dist):.3f} ± {std(dist):.3f}')

    print("DISTANCE FROM EMPIRICAL DISTRIBUTION")
    print('Method\t\tBlock\t\tLine')
    for m, dist in block_dist.items():
        if m == "emp": continue
        b = wasserstein_distance(dist, block_dist["emp"])
        l = wasserstein_distance(dist, block_dist["emp"])
        print(f'\t{m}\t\t{b:.3f}\t\t{l:.3f}')


if __name__ == '__main__':
    merging_distributions()