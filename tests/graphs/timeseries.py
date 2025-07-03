import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean, std, array
from tests.data import merging_sc

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
    sims = [m_dict["sims"] for m_dict in merging_sc.values()]
    labels = ['Συμβολοσειρά', 'Περιστροφή', 'Απομάκρυνση', 'Τυχαία', 'Εξαντλητική'] if lang == 'gr' \
        else ['String', 'Rotation', 'Digression', 'Random', 'Exhaustive']
    title = 'Ομοιότητα Αποτελέσματος-Προδιαγραφής σε Δείγματα' if lang == 'gr' \
        else 'Query-Result Similarity over Block Samples'
    xlabel = 'Αριθμός Δειγμάτων Μπλοκ' if lang == 'gr' else 'Block Sample Size'
    ylabel = 'Μέση Συνημιτονική Ομοιότητα' if lang == 'gr' else 'Mean Cosine Similarity'
    path = '../../plots/merging_ts.png'
    plot_tsplot(labels, sims, title, xlabel, ylabel, path)


if __name__ == "__main__":
    merging_ts()