from numpy import mean, std
from tests.data.merging_quan import merging_quan
from tests.data.merging_run import llama
from tests.data.merging_run import synthesis
from tests.data.preprocessing_sc import preprocessing_sc

def merging_run():
    print(f'Synthesis Merge Execution Time Range: {mean(synthesis):.2f} ± {std(synthesis)} s')
    print(f'Llama Generation Execution Time Range: {mean(llama):.2f} ± {std(llama):.2f} s')


def preprocessing():
    print("Number of Blocks Processed\t\tExecution Time Range")
    for b, pr_times in preprocessing_sc.items():
        print(f'{b}:\t\t{mean(pr_times):.2f} ± {std(pr_times):.2f}s')


def merging_quantitative():
    print('Method\t\tSimilarity Range:')
    for m, m_dict in merging_quan.items():
        print(f'{m}\t\t{mean(m_dict["sims"]):.2f} ± {std(m_dict["sims"]):.2f}')

    print('Method\t\tBlock Range:')
    for m, m_dict in merging_quan.items():
        print(f'{m}\t\t{mean(m_dict["blocks"])} ± {std(m_dict["blocks"])}')
    print('Method\t\tLines Range:')
    for m, m_dict in merging_quan.items():
        print(f'\t{m}: {mean(m_dict["lines"])} ± {std(m_dict["lines"])}')


if __name__ == '__main__':
    merging_quantitative()