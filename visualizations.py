import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from wordle_game import experiment_name


def parse_experiment_files(experiments: List[Dict]) -> Dict[str, List]:
    n_experiments = len(experiments)
    table = {}
    for i, experiment in enumerate(experiments):
        file = experiment_name(**experiment)
        with open(file, 'r') as f:
            data = json.load(f)
        scores = data['scores']
        mean_scores = {w: np.mean([x for x in s if x > -1]) for w, s in scores.items()}
        for word, mean_score in mean_scores.items():
            if word not in table:
                table[word] = [-1 for _ in range(n_experiments)]
            table[word][i] = mean_score
    return table


def latex_table(experiments: List[Dict]) -> str:
    table = parse_experiment_files(experiments)

    res = '\\begin{tabular}{' + '|c|' + 'c' * len(experiments) + '|}\n'
    res += '\\hline\n'
    res += ' & '.join(['\\textbf{Opening word}']
                      + ['\\textbf{' + header_name(experiment) + '}' for experiment in experiments]) + '\n'
    res += '\\hline\n'

    for word, mean_scores in table.items():
        res += ' & '.join([word] + [f'{x:.4f}' for x in mean_scores]) + '\n'

    res += '\\hline\n'
    res += '\\end{tabular}'
    return res


def bar_chart(experiments: List[Dict], outfile: str = 'out/mean-scores.svg') -> None:
    bar_width = 0.25
    table = parse_experiment_files(experiments)
    opening_words = table.keys()
    n_opening_words = len(opening_words)
    n_experiments = len(experiments)

    plt.figure()
    br = np.arange(n_opening_words)
    for i, experiment in enumerate(experiments):
        plt.bar(br, [table[w][i] for w in opening_words], width=bar_width, label=header_name(experiment))
        br = [x + bar_width for x in br]

    plt.xlabel('Opening word')
    plt.ylabel('Mean # of guesses')
    plt.xticks([r + bar_width * (1/2 * (n_experiments / 2)) for r in range(n_opening_words)], opening_words)
    plt.ylim(3.5, None)
    plt.legend()

    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)


def header_name(experiment: Dict) -> str:
    return f'{"Rollout with " if experiment["rollout"] else ""}MIG as base heuristic'


if __name__ == '__main__':
    experiments = [
        {
            'rollout': False,
        },
        {
            'rollout': True,
            'n_samples': 20,
            'top_k': 3,
        },
    ]
    print(latex_table(experiments))
    bar_chart(experiments)
