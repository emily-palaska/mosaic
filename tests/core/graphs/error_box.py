import seaborn as sns
import matplotlib.pyplot as plt
from numpy import mean, std
from tests.core.utils import linear_regression

colors = ['#309db1', '#1f497d']

def plot_boxplot(labels, values, title='Boxplot', x='Label', y='Value', path=None):
    flat_l = [l for l, v_list in zip(labels, values) for _ in v_list]
    flat_v = [v for v_list in values for v in v_list]
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=flat_l, y=flat_v, palette=colors, showfliers=False, showmeans=True, meanline=True,
                meanprops={'linestyle': '-', 'color': 'black'})
    for color, v_list in zip(colors, values):
        label = rf'$\mu \pm \sigma$ = {mean(v_list):.2f} $\pm$ {std(v_list):.2f} ms'
        plt.plot([], [], color=color, marker='s', linestyle='', label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if path: plt.savefig(path)
    else: plt.show()
    plt.close()


def plot_errorbar(times, title, xlabel, ylabel, llabel, path=None):
    flat_x = [block for block, t_list in times.items() for _ in t_list]
    flat_y = [t for t_list in times.values() for t in t_list]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.pointplot(x=flat_x, y=flat_y, capsize=0.2, errorbar="sd", palette="YlGnBu_d", markers='o', markersize=4.0,
        linewidth=2.0)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    y_pred, m, b, rs = linear_regression(list(times.keys()), [mean(t_list) for t_list in times.values()])
    ax.plot(ax.get_xticks(), y_pred, color='lightseagreen', linestyle='--', label=f'{llabel}: $R^2$ = {rs:.2f}',
            linewidth=3.0)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if path: plt.savefig(path)
    else: plt.show()
    plt.close()


def retrieving_boxplot(lang='gr'):
    from tests.core.graphs.data import exactnn, approxnn
    for i in range(len(exactnn)): exactnn[i], approxnn[i] = 1000 * exactnn[i], 1000 * approxnn[i]
    labels = ['Ακριβής', 'Προσεγγιστικός'] if lang == 'gr' else ['Exact', 'Approx']
    title = f'Θηκόγραμμα Χρόνου Ανάκτησης {len(exactnn)} Επαναλήψεων' if lang == 'gr'\
        else f'Retrieving Time Boxplot for {len(exactnn)} Recurrences'
    xlabel ='Αλγόριθμος Κοντινότερου Γείτονα' if lang == 'gr' else f'Nearest Neighbor Algorithm'
    ylabel = 'Χρόνος Εκτέλεσης (ms)' if lang == 'gr' else 'Execution Time (ms)'
    path = f'../../../plots/retrieving_box.png'
    plot_boxplot(labels, [exactnn, approxnn], title, xlabel, ylabel, path)


def preprocessing_errorbar(lang='gr'):
    from tests.core.graphs.data import preprocessing_sc as times
    per_block = [v / int(k) for k, values in times.items() for v in values]
    print(f'Processing of one block: {mean(per_block):.2f} ± {std(per_block):.2f} s')
    xlabel = 'Αριθμός Μπλοκ' if lang == 'gr' else 'Block Number'
    ylabel = 'Χρόνος (s)' if lang == 'gr' else 'Processing Time (s)'
    llabel = 'Ευθεία Ελαχίστων Τετραγώνων' if lang == 'gr' else 'Least Squares Line'
    title = 'Χρόνος Επεξεργασίας Μπλοκ' if lang == 'gr' else 'Block Processing Time'
    path = f'../../../plots/preprocessing_error.png'
    plot_errorbar(times, title, xlabel, ylabel, llabel, path)


def merging_calculations():
    from tests.core.graphs.data import synthesis, llama
    print(f'Synthesis: {mean(synthesis):.2f} ± {std(synthesis)} ms')
    print(f'Llama: {mean(llama):.2f} ± {std(llama):.2f} ms')

if __name__ == '__main__':
    merging_calculations()
    preprocessing_errorbar()
    retrieving_boxplot()
