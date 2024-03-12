import copy
import json
import os.path
from functools import cached_property
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import entropy

from wordle_run import Color, WordleRun, color_word


class WordleGame:
    """Wordle simulator/solver.

    Attributes:
        guess_list (List[str]): List of allowed words to guess.
        mystery_list (List[str]): List of potential solutions/mystery words (subset of guess_list).
        top_k (int): Perform rollouts for top_k words in lookahead.
        priors (np.ndarray): Prior on being solution/mystery word for each word in guess_list.
        verbose (bool): True if solving functions should be verbose.
    """
    def __init__(
            self,
            guess_list: Optional[str] = None,
            mystery_list: Optional[str] = None,
            top_k: int = 10,
            verbose: bool = True,
    ) -> None:
        """Initialize Wordle simulator.

        Args:
            guess_list: List of allowed words to guess.
            mystery_list: List of potential mystery words.
            top_k: Perform rollouts for top_k words in lookahead.
            verbose: True if solving functions should be verbose.
        """
        if guess_list is None:
            guess_list = 'word-lists/guess-standard.txt'
        if mystery_list is None:
            mystery_list = 'word-lists/mystery-standard.txt'

        self.guess_list = parse_word_list(guess_list)
        self.mystery_list = parse_word_list(mystery_list)
        assert len(set(map(len, self.mystery_list))) == 1  # all words the same length
        assert set(self.mystery_list).issubset(set(self.guess_list))

        self.top_k = top_k
        self.verbose = verbose

        self.priors = np.array([1. if word in self.mystery_list else 0. for word in self.guess_list])
        self.priors /= self.priors.sum()

    @cached_property
    def len_word(self) -> int:
        return len(self.guess_list[0])

    @cached_property
    def n_guess(self) -> int:
        return len(self.guess_list)

    @cached_property
    def n_mystery(self) -> int:
        return len(self.mystery_list)

    @cached_property
    def word_ind_map(self) -> Dict[str, int]:
        return {word: i for i, word in enumerate(self.guess_list)}

    def random_mystery_word(self) -> str:
        return np.random.choice(self.mystery_list)

    def play(self, show_mystery_word: bool = True) -> None:
        """Play interactively."""
        WordleRun(self.random_mystery_word()).cli(show_mystery_word=show_mystery_word)

    def one_step_lookahead_minimization(
            self, run: WordleRun,
            priors: np.ndarray,
            rollout: bool = True,
            n_samples: int = -1
    ) -> str:
        """Implements Algorithm 1 of Bhambri et al.

        Args:
            run: Contains sequence of previous guesses before this step and mystery word.
            priors: Priors on being solution for each word in self.guess_list.
            rollout: If True, uses one step lookahead with rollouts, else just uses base heuristic.
            n_samples: If > 0, randomly sample this number of mystery words per rollout instead of
                using the full list of mystery words.

        Returns:
            Optimal word to guess.
        """
        if self.verbose:
            print(f'====> Starting one step lookahead minimization {"w/ rollouts " if rollout else ""}'
                  f'with current board:')
            print(run)

        # mystery_list = [word for i, word in enumerate(self.guess_list) if priors[i] > 0]
        ents, mystery_list = self.get_entropy_scores(run, priors)
        max_indices = np.argsort(ents)[::-1][:self.top_k]
        if ents[max_indices[0]] == 0:
            if self.verbose:
                print('Entropy all 0, setting top choice to mystery word')
            return mystery_list[0]
        top_choices = [self.guess_list[idx] for idx in max_indices]
        if not rollout:
            return top_choices[0]

        if self.verbose:
            print(f'Top choices: {top_choices}')

        top_choices_Q = []
        for i, choice in enumerate(top_choices):
            choice_q_factors = []

            if 0 < n_samples < len(mystery_list):
                weights = np.array([priors[self.word_ind_map[m]] for m in mystery_list])
                weights /= sum(weights)
                mystery_list_samples = np.random.choice(mystery_list, size=n_samples, replace=False, p=weights)
                if self.verbose:
                    print(f'Using {n_samples} mystery samples for rollouts: {mystery_list_samples}')
            else:
                mystery_list_samples = mystery_list

            for j, m in enumerate(mystery_list_samples):
                if self.verbose:
                    print(f'====> Rollout for choice: {choice} ({i + 1}/{self.top_k}) | '
                          f'mystery word: {m} ({j + 1}/{len(mystery_list_samples)})')
                guess = choice
                score = 0
                run_ = copy.deepcopy(run)
                run_.mystery_word = m
                run_.make_guess(choice)
                if self.verbose:
                    print(f'Made guess: {guess}')
                    print(run_)
                while guess is not m:
                    # if self.verbose:
                    #     print(f'Guessing {guess}...')
                    guess = self.guess_max_information_gain_word(run_, priors)
                    if self.verbose:
                        print(run_)
                    score += 1
                choice_q_factors.append(score)
            mean_q = 1. / len(mystery_list) * sum(choice_q_factors)
            top_choices_Q.append(mean_q)
        min_index = np.argmin(top_choices_Q)
        return top_choices[min_index]

    def solve_run(
            self,
            opening_word: str,
            mystery_word: Optional[str] = None,
            rollout: bool = True,
            n_samples: int = -1
    ) -> WordleRun:
        """Start a new playthrough of the game and attempt to solve."""
        if mystery_word is None:
            mystery_word = self.random_mystery_word()
        run = WordleRun(mystery_word)
        print(run)
        score = 0
        guess = opening_word
        run.make_guess(guess)
        print(run)
        priors = self.priors
        while guess != run.mystery_word:
            guess = self.one_step_lookahead_minimization(run, priors, rollout=rollout, n_samples=n_samples)
            run.make_guess(guess)
            # priors = self.get_posteriors(run, priors)
            score += 1
            print(run)
        return run

    def simulate_runs(
            self,
            opening_words: Optional[List[str]] = None,
            save_file: Optional[str] = None,
            rollout: bool = True,
            n_samples: int = 20,
            n_new_sims: int = 100,
            src_mystery_words: Optional[str] = None,
    ) -> Dict[str, float]:
        """Simulates runs over different mystery words to gather resulting scores.

        Writes resulting score data to json file save_file, in format
        {"mystery_list": [mystery words whose scores were computed],
        "scores": {opening word: corresponding score for mystery word at this index}}.

        Args:
            opening_words: List of opening words to simulate runs from.
            save_file: Path to save score data from simulations.
            rollout: If True, uses one step lookahead with rollouts, else just uses base heuristic.
            n_samples: Number of mystery words samples to use in rollouts for lookahead.
            n_new_sims: Number of additional random mystery words to generate and run experiments for.
            src_mystery_words: Use the mystery words from this json file (representing a set of simulations).

        Returns:
            Mean scores for each of the opening words.
        """
        if opening_words is None:
            opening_words = [
                'salet',
                'reast',
                'crate',
                'trace',
                'slate',
                # 'trape',
                # 'slane',
                # 'prate',
                # 'crane',
                # 'carle',
                # 'train',
                # 'raise',
                # 'clout',
            ]

        if save_file is None:
            save_file = experiment_name(rollout=rollout, n_samples=n_samples, top_k=self.top_k)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        mystery_list = []
        scores = {}
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                data_ = json.load(f)
            mystery_list = data_["mystery_list"]
            scores = data_["scores"]

        if src_mystery_words is not None:
            with open(src_mystery_words, 'r') as f:
                data_ = json.load(f)
            new_mystery_words = data_["mystery_list"]
            new_mystery_words = set(new_mystery_words).difference(mystery_list)
            mystery_list += new_mystery_words

        new_mystery_words = np.random.choice(self.guess_list, size=n_new_sims, replace=False, p=self.priors)
        new_mystery_words = set(new_mystery_words).difference(mystery_list)
        # print(new_mystery_words)
        mystery_list += new_mystery_words
        assert all([word in self.mystery_list for word in mystery_list])

        n_mystery_list = len(mystery_list)
        for opening_word in opening_words:
            if opening_word not in scores:
                scores[opening_word] = []
            scores[opening_word] += [-1 for _ in range(n_mystery_list - len(scores[opening_word]))]
        # scores = dict(zip(opening_words, [scores[word] + [-1 for _ in range(len(mystery_list))] for i, word in enumerate(opening_words)]))

        n_opening_words = len(opening_words)
        for i, mystery_word in enumerate(mystery_list):
            for k, opening_word in enumerate(opening_words):
                if scores[opening_word][i] > -1:
                    continue
                print(f'===> Solving for opening word: {opening_word} ({k + 1}/{n_opening_words}) | '
                      f'mystery word: {mystery_word} ({i + 1}/{n_mystery_list})')
                run = self.solve_run(opening_word, mystery_word=mystery_word, rollout=rollout, n_samples=n_samples)
                score = run.score
                scores[opening_word][i] = score
                with open(save_file, 'w') as f:
                    data = {'mystery_list': mystery_list, 'scores': scores}
                    json.dump(data, f, indent=2)
        mean_scores = {word: sum(scores[word]) / len(scores[word]) for word in scores}
        return mean_scores

    @cached_property
    def pattern_matrix(
            self,
            recompute: bool = False,
            file: str = 'out/pattern_matrix.npy',
    ) -> np.ndarray:
        """Stores pairwise color patterns for each pair of words in guess list.

        self.pattern_matrix[a][b] gives the pattern of if self.guess_list[a]
        were the guess word and self.guess_list[b] were the mystery word.
        A pattern for two words represents the wordle-similarity
        pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
        between 0 and 3^5. Reading this integer in ternary gives the
        associated pattern. Computation is time-intensive so stores to file to
        load/lookup in future runs.

        Args:
            recompute: If true, don't read from file and recompute the matrix.
            file: Path to save

        Returns:
            Matrix of color patterns for each pair of words.
        """
        if not recompute and os.path.exists(file):
            if self.verbose:
                print(f'Loading pattern matrix: {file}.')
            return np.load(file)

        # full_pattern_matrix[a, b] should represent the 5-color pattern
        #  for guess a and answer b
        pattern_grid = np.zeros((self.n_guess, self.n_guess, self.len_word), dtype=np.uint8)

        if self.verbose:
            print('Computing pattern matrix...')
        for a in range(self.n_guess):
            if a % 100 == 0 and self.verbose:
                print(f'{a}/{self.n_guess} words')
            for b in range(self.n_guess):
                pattern_grid[a][b][:] = list(map(int, color_word(self.guess_list[a], self.guess_list[b])))

        # Rather than representing a color pattern as a lists of integers,
        #  store it as a single integer, whose ternary representations corresponds
        #  to that list of integers.
        pattern_grid = np.dot(pattern_grid, 3 ** np.arange(self.len_word).astype(np.uint8))

        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        np.save(file, pattern_grid)
        if self.verbose:
            print(f'Saved pattern matrix: {file}.')

        return pattern_grid

    def get_posteriors(self, run: WordleRun, priors: np.ndarray) -> (np.ndarray, List[str]):
        """Updated probabilities based on run history and priors.

        Returns:
            Posterior probabilities, list of remaining possible mystery words.
        """
        posterior = priors.copy()
        # condensed_history = run.condensed_history()
        n_eliminated = 0
        for word in self.mystery_list:
            i = self.word_ind_map[word]
            if not self.is_possible_solution(run, word):
            # if not is_possible_solution(word, condensed_history):
                posterior[i] = 0.
                n_eliminated += 1
        mystery_list = [word for word in self.mystery_list if posterior[self.word_ind_map[word]] > 0.]
        if self.verbose:
            print(f'{self.n_mystery - n_eliminated}/{self.n_mystery} possible mystery words:')
            print(mystery_list)
            # print(len(mystery_list))
        total = posterior.sum()
        if total == 0:
            return np.zeros(posterior.shape)
        else:
            posterior /= posterior.sum()
            return posterior, mystery_list

    def get_entropy_scores(self, run: WordleRun, priors: np.ndarray) -> (np.ndarray, List[str]):
        """Entropy for each word in self.guess_list given run history and priors.

        Returns:
            Entropies, list of remaining possible mystery words.
        """
        weights, mystery_list = self.get_posteriors(run, priors)

        mystery_inds = [self.word_ind_map[w] for w in mystery_list]
        pattern_matrix = self.pattern_matrix[:, mystery_inds]
        weights = weights[np.nonzero(weights)]
        weights /= sum(weights)
        assert len(weights) == len(mystery_list)

        distributions = np.zeros((self.n_guess, 3 ** 5))
        n_range = np.arange(self.n_guess)
        for i, prob in enumerate(weights):
            distributions[n_range, pattern_matrix[:, i]] += prob
        entropies = entropy(distributions, base=2, axis=-1)
        return entropies, mystery_list

    def guess_max_information_gain_word(self, run: WordleRun, priors: np.ndarray) -> str:
        """Guesses the MIG (highest-entropy) word for run (adds to run.guess_history).

        Returns:
            Word guessed.
        """
        ents, mystery_list = self.get_entropy_scores(run, priors)
        # print(sorted(ents)[::-1][:10])
        max_ent_idx = np.argmax(ents)
        if ents[max_ent_idx] == 0:
            run.make_guess(run.mystery_word)
            if self.verbose:
                print(f'Made guess: {run.mystery_word}')
            return run.mystery_word  # just cheat here since we know the word
        mig_word = self.guess_list[np.argmax(ents)]
        run.make_guess(mig_word)
        if self.verbose:
            print(f'Made guess: {mig_word}')
        return mig_word

    def is_possible_solution(self, run: WordleRun, word: str) -> bool:
        """Returns True if word is a possible solution given run history."""
        word_idx = self.word_ind_map[word]
        board = np.array(run.board())
        board = np.dot(board, 3 ** np.arange(self.len_word).astype(np.uint8))
        for i, guess in enumerate(run.guess_history):
            guess_idx = self.word_ind_map[guess]
            if self.pattern_matrix[guess_idx][word_idx] - board[i] != 0:
                return False
        return True


def parse_word_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f.read().strip().splitlines()]


def experiment_name(rollout: bool = True, n_samples: int = -1, top_k: int = -1):
    s = f'out/scores-{"rollout" if rollout else "base"}'
    if rollout:
        s += f'_n-samples-{n_samples}_top-k-{top_k}'
    s += '.json'
    return s


if __name__ == '__main__':
    game = WordleGame(top_k=3, verbose=False)
    # res = game.simulate_runs(rollout=True, n_samples=20, n_new_sims=0)
    res = game.simulate_runs(rollout=False, n_new_sims=0,
                             src_mystery_words=experiment_name(rollout=True, n_samples=20, top_k=3))
    print(res)
