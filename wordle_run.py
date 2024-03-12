import colorama
from colorama import Back, Fore, Style
from enum import IntEnum, auto
from functools import cached_property
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class Color(IntEnum):
    GRAY = 0
    YELLOW = 1
    GREEN = 2


class WordleRun:
    """A single run/playthrough of a Wordle game.

    Attributes:
        mystery_word (str): If you guess this word, you win!
        max_guesses (int): Number of guesses before the game is lost.
        hard_mode (bool): If True, turn on hard mode (must preserve yellow and green letters of previous guess).
        guess_history (List[str]): History of guesses made in this game.
    """
    def __init__(
            self,
            mystery_word: str,
            max_guesses: int = 6,
            hard_mode: bool = False,
            guess_history: Optional[List[str]] = None
    ) -> None:
        """Initialize Wordle game.

        Args:
            max_guesses: Number of guesses before the game is lost.
            hard_mode: If True, use hard mode.
            mystery_word: The hidden mystery word. If None, one is selected randomly from the mystery list.
        """
        self.mystery_word = mystery_word
        self.max_guesses = max_guesses
        self.hard_mode = hard_mode
        if guess_history is None:
            self.guess_history = []
        else:
            self.guess_history = guess_history

        colorama.init()

    @cached_property
    def len_word(self):
        return len(self.mystery_word)

    @property
    def score(self):
        return len(self.guess_history)

    def guesses_remaining(self) -> int:
        return self.max_guesses - len(self.guess_history)

    def make_guess(self, word: str) -> None:
        """Add a word to guess history if it is valid guess.

        A 'valid' guess fulfills the rules of hard mode if self.hard_mode.
        Guesses are still 'valid' if more than self.max_guesses has been exceeded.

        Args:
            word: Word to guess.

        Returns:
            True if guess was valid, otherwise False.
        """
        if len(word) != self.len_word:
            raise ValueError

        if self.hard_mode:
            prev_guess = self.guess_history[-1]
            prev_colors = self.color_word(prev_guess)
            for i, prev_color in enumerate(prev_colors):
                if prev_color == Color.GREEN and word[i] != prev_guess[i]:
                    raise ValueError
                elif prev_color == Color.YELLOW and word[i] not in prev_guess[i]:
                    raise ValueError

        self.guess_history.append(word)

    def color_word(self, word: str) -> List[Color]:
        """Returns letter colors for a given word."""
        return color_word(word, self.mystery_word)

    def board(self) -> List[List[Color]]:
        """Returns colored letter grid from guess history."""
        return [self.color_word(guess) for guess in self.guess_history]

    def board_repr(self) -> str:
        """Returns string representation of board for pretty-printing w/ colors."""
        board = self.board()
        colors = {Color.GREEN: Fore.GREEN, Color.YELLOW: Fore.YELLOW, Color.GRAY: Style.RESET_ALL}
        lines = []
        for i, word in enumerate(self.guess_history):
            line = [f'{colors[board[i][j]]}{letter}{Style.RESET_ALL}' for j, letter in enumerate(word)]
            lines.append(' '.join(line))
        lines_remaining = self.max_guesses - len(lines)
        if lines_remaining > 0:
            lines += [' '.join(['-' for _ in range(self.len_word)]) for _ in range(lines_remaining)]
        return '\n'.join(lines)

    def __str__(self) -> str:
        return f"[Mystery word: {self.mystery_word}]\n{self.board_repr()}"

    def cli(self, show_mystery_word: bool = False) -> None:
        """Play interactively.

        Args:
            show_mystery_word: If True, show mystery word to player before the game.
        """
        if show_mystery_word:
            print(f'Mystery word: {self.mystery_word}')
        print(self.board_repr())

        while self.guesses_remaining() > 0:
            word = input('Guess a word: ')
            if not self.make_guess(word):
                print(f'Not a valid guess: {word}; try again.')
            print(self.board_repr())
            if word == self.mystery_word:
                print(f'Correct, mystery word: {self.mystery_word}.')
                return
        print(f'You ran out of guesses :(, mystery word: {self.mystery_word}.')


def color_word(guess_word: str, mystery_word: str) -> List[Color]:
    """Returns letter colors for a given guess and mystery word."""
    assert len(guess_word) == len(mystery_word)
    colors = [Color.GRAY for _ in range(len(guess_word))]
    for i, letter in enumerate(guess_word):
        if letter in mystery_word and mystery_word[i] == letter:
            colors[i] = Color.GREEN
        elif letter in mystery_word:
            colors[i] = Color.YELLOW
    return colors
