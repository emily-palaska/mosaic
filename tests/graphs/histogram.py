import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def plot_histogram(data, bins=20, title='Histogram', x='Value', y='', l='', path=None):
    sns.histplot(data, bins=bins, kde=True, color='#309db1', label=l)
    mean = np.mean(data)
    std = np.std(data)
    label = rf'$\mu \pm \sigma$ = {mean:.2f} $\pm$ {std:.2f} s'
    plt.axvline(mean, color='#1f497d', linestyle='--', linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    if path: plt.savefig(path)
    else: plt.show()
    plt.close()


def preprocess_hist(lang='gr'):
    from tests.graphs.data import preprocessing_run as times
    samples = len(times)
    x_label = 'Χρόνος Εκτέλεσης (s)' if lang == 'gr' else 'Execution Time (s)'
    title = f'Χρόνος Eπεξεργασίας 31 Μπλοκ' if lang == 'gr' \
        else f'<Processing 31 Blocks Time Histogram'
    l = f'{samples} Επαναλήψεις' if lang == 'gr' else f'{samples} Recurrences'
    path = '../../plots/preprocessing_hist.png'
    plot_histogram(times, title=title, x=x_label, l=l, path=path)


def restore_hist(lang='gr'):
    from tests.graphs.data import restoring_run as times
    samples = len(times)
    x_label = 'Χρόνος Εκτέλεσης (s)' if lang == 'gr' else 'Execution Time (s)'
    title = f'Χρόνος Φόρτωσης 31 Επεξεργασμένων Μπλοκ' if lang == 'gr' \
        else f'<Restoring 31 Pre-processed Blocks Time Histogram'
    l = f'{samples} Επαναλήψεις' if lang == 'gr' else f'{samples} Recurrences'
    path = '../../plots/restoring_hist.png'
    plot_histogram(times, title=title, x=x_label, l=l, path=path)


if __name__ == '__main__':
    preprocess_hist()
    restore_hist()
