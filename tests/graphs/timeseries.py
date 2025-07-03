import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean, std, array

def plot_tsplot(labels, values, title='TSPlot', x='Block Sample Size', y='Value', path=None):
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("YlGnBu_d") if len(labels) >= 3 else sns.color_palette("Set2")
    palette = sns.color_palette("bright")
    for label, block_data, color in zip(labels, values, palette):
        xs = sorted(block_data.keys())
        ys_mean, ys_std = [mean(block_data[x]) for x in xs], [std(block_data[x]) for x in xs]

        sns.lineplot(x=xs, y=ys_mean, label=label, color=color)
        plt.fill_between(xs,
                         array(ys_mean) - array(ys_std),
                         array(ys_mean) + array(ys_std),
                         alpha=0.2, color=color)

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    if path: plt.savefig(path)
    else: plt.show()
    plt.close()

def merging_ts(lang='gr'):
    from tests.graphs.merging_sc import merging_sc
    sims = [m_dict["sims"] for m_dict in merging_sc.values()]

    labels = ['Συμβολοσειρά', 'Περιστροφή', 'Απομάκρυνση', 'Τυχαία', 'Εξαντλητική'] if lang == 'gr' \
        else ['String', 'Rotation', 'Digression', 'Random', 'Exhaustive']
    title = 'Ομοιότητα Αποτελέσματος-Προδιαγραφής σε Δείγματα' if lang == 'gr' \
        else 'Query-Result Similarity over Block Samples'
    xlabel = 'Αριθμός Δειγμάτων Μπλοκ' if lang == 'gr' else 'Block Sample Size'
    ylabel = 'Μέση Συνημιτονική Ομοιότητα' if lang == 'gr' else 'Mean Cosine Similarity'
    path = '../../plots/merging_ts.png'

    plot_tsplot(labels, sims, title, xlabel, ylabel, path)

def merging_calculations():
    from tests.graphs.merging_quan import merging_quan
    from tests.queries import queries
    stopwords = [
    "a", "an", "the",
    "and", "or", "but", "if", "because", "while",
    "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "up", "down", "over", "under", "above", "below",
    "as", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "it", "its", "they", "them", "their",
    "will", "would", "can", "could", "shall", "should",
    "when", "where", "how", "why", "what", "which",
    "not", "no", "yes", "so", "such",
    "there", "here", "other", "another",
    "more", "most", "less", "least",
    "only", "just", "also", "very", "too", "much", "many", "some", "any",
    "each", "every", "all", "both", "either", "neither"
]
    filtered_queries = []
    for q in queries:
        for word in stopwords:
            q = q.replace(word, '')
        filtered_queries.append(q)

    assert len(filtered_queries) == len(queries)
    block_counts = {method: m_dict["blocks"] for method, m_dict in merging_quan.items()}
    line_counts = {method: m_dict["lines"] for method, m_dict in merging_quan.items()}



    q_words = [len(q.split()) for q in filtered_queries]
    assert len(block_counts["e"]) == len(q_words)
    assert len(line_counts["e"]) == len(q_words)

    print(f"Average Words per Query: {mean(q_words)} ± {std(q_words)}")
    print('Average Blocks per Method:')
    for m, bc in block_counts.items(): print(f'\t{m}: {mean(bc)} +- {std(bc)}')
    print('Average Lines per Method:')
    for m, lc in line_counts.items(): print(f'\t{m}: {mean(lc)} +- {std(lc)}')

    print("Block count / Query words for each Query then Average per Method:")
    blocks_per_q_words = {m: [bc / qw for bc, qw in zip(block_count, q_words)] for m, block_count in
                          block_counts.items()}
    for method, count in blocks_per_q_words.items(): print(f'\t{method}: {mean(count)} +- {std(count)}')

    print("Line count / Query words for each Query then Average per Method:")
    lines_per_q_words = {m: [lc / qw for lc, qw in zip(line_count, q_words)] for m, line_count in
                          line_counts.items()}
    for method, count in lines_per_q_words.items(): print(f'\t{method}: {mean(count)} +- {std(count)}')

    for method, line_count in lines_per_q_words.items():
        plt.figure()
        plt.hist(line_count, bins=20)
        plt.title("Line/Query - " + method)
        plt.savefig(f'{method}.png')
        plt.close()

if __name__ == "__main__":
    merging_calculations()
    merging_ts()