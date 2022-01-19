from abc import ABC, abstractmethod
from mancala import Mancala
from random import choice
from time import sleep


class Agent(ABC):

    @abstractmethod
    def move(self, game: Mancala):
        pass

    @staticmethod
    def get_possible_moves(game: Mancala) -> list[int]:
        """Return a list of indices for possible game moves"""
        indices = list(game._p1_range) if game.p1 else list(game._p2_range)

        # return [index for index in indices if game.pits[index] > 0]
        return list(filter(lambda index: game.pits[index] > 0, indices))


class AgentRandom(Agent):
    """Randomly select any pit that contains at least 1 seed"""

    def move(self, game: Mancala) -> int:
        possible_moves = self.get_possible_moves(game=game)
        sleep(0.5)
        return choice(possible_moves)


class AgentManual(Agent):
    """Human player, manually selects a pit"""

    def move(self, game: Mancala) -> int:
        return input()


class AgentMinimax(Agent):

    def move():
        pass


class AgentMinimaxAlphaBeta(Agent):

    def move():
        pass
