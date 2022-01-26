from abc import ABC, abstractmethod
from mancala import Mancala
from random import choice
from time import sleep


class Agent(ABC):

    @abstractmethod
    def move(self, game: Mancala):
        pass

    @staticmethod
    def evaluation(game: Mancala, player: int) -> int:
        """Return heuristic value of the terminal node"""
        # return game.game_score[player]
        if game.game_over:
            return 100 if game.game_winner == player else -100
        return game.game_score[player] - game.game_score[not player]

    @staticmethod
    def get_possible_moves(game: Mancala) -> list[int]:
        """Return a list of indices for possible game moves"""
        indices = list(game._p1_range) if game.p1 else list(game._p2_range)

        # return [index for index in indices if game.pits[index] > 0]
        return list(filter(lambda index: game.pits[index] > 0, indices))


class AgentRandom(Agent):
    """Randomly select any pit that contains at least 1 seed"""

    @staticmethod
    def move(game: Mancala) -> int:
        possible_moves = Agent.get_possible_moves(game=game)
        sleep(0.5)
        return choice(possible_moves)


class AgentManual(Agent):
    """Human player, manually selects a pit"""

    def move(self, game: Mancala = None) -> int:
        return input()


class AgentMinimax(Agent):

    def __init__(self, p1: int, depth: int = 3):
        self._p1 = p1
        self._depth = depth

    @property
    def p1(self) -> int:
        """Return 1 (True) if Agent is Player 1, else 0 (False)"""
        return self._p1

    @property
    def depth(self) -> int:
        return self._depth

    @staticmethod
    def minimax(depth: int, game: Mancala, pit: int) -> int:
        """Recursively traverse the game tree and select a pit that minimizes the maximum loss"""

        clone = game.clone()
        score_diff, capture, go_again = clone.sow(pit=pit)
        maximizer = clone.p1

        if depth == 0:
            return Agent.evaluation(game=clone, player=maximizer)

        best_move = -float("inf") if maximizer else float("inf")
        possible_moves = Agent.get_possible_moves(game=clone)

        for move in possible_moves:
            move_value = AgentMinimax.minimax(depth=depth-1, game=clone, pit=move)
            best_move = max(best_move, move_value) if maximizer else min(best_move, move_value)

        return best_move

    def move(self, game: Mancala) -> int:

        clone = game.clone()
        possible_moves = Agent.get_possible_moves(game=clone)
        possible_scores = [AgentMinimax.minimax(self.depth, clone, pit) for pit in possible_moves]

        best_score = max(possible_scores)
        best_pits = [pit for score, pit in zip(possible_scores, possible_moves) if score == best_score]  # noqa
        return choice(best_pits)


class AgentMinimaxAlphaBeta(AgentMinimax):

    @staticmethod
    def minimax(depth: int, game: Mancala, pit: int, alpha: int, beta: int) -> int:
        """Recursively traverse the game tree and select a pit that minimizes the maximum loss"""

        clone = game.clone()
        score_diff, capture, go_again = clone.sow(pit=pit)
        maximizer = clone.p1

        if depth == 0:
            return Agent.evaluation(game=clone, player=maximizer)

        best_move = -float("inf") if maximizer else float("inf")
        possible_moves = Agent.get_possible_moves(game=clone)

        for move in possible_moves:
            move_value = AgentMinimax.minimax(depth=depth-1, game=clone, pit=move)
            best_move = max(best_move, move_value) if maximizer else min(best_move, move_value)

            if maximizer:
                alpha = max(alpha, best_move)
            else:
                beta = min(beta, best_move)

            if beta <= alpha:
                return best_move

        return best_move

    def move(self, game: Mancala) -> int:

        clone = game.clone()
        possible_moves = Agent.get_possible_moves(game=clone)
        possible_scores = [AgentMinimaxAlphaBeta.minimax(
            self.depth, clone, pit, -float("inf"), float("inf")) for pit
            in possible_moves]

        best_score = max(possible_scores)
        best_pits = [pit for score, pit in zip(possible_scores, possible_moves) if score == best_score]  # noqa
        return choice(best_pits)

