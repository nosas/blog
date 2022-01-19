from mancala_errors import IllegalMoveError


class Mancala:

    _pits_default = [4, 4, 4, 4, 4, 4, 0,
                     4, 4, 4, 4, 4, 4, 0]
    _mancala_idxs = [6, 13]  # P2 mancala == 6, P1 mancala == 13
    _pit_idxs = range(0, 13)
    _p1_range = range(0, 6)
    _p2_range = range(7, 13)

    def __init__(self, pits: list[int] = None):
        """Create a new game of Mancala

        If no value is passed to `pits` argument, the default starting pit values are used

        Args:
            pits (list[int]): List of 14 integers representing the pits and mancalas.
                                Refer to example render for expected indices of pits and mancalas.

                :: Expected pit/mancala indices
                      0   1   2   3   4   5
                |===============================|     13 = Player1's mancala
                |---| 4 | 4 | 4 | 4 | 4 | 4 |---|    0-5 = Player1's pits
            13  | 0 |=======================| 0 | 6
                |---| 4 | 4 | 4 | 4 | 4 | 4 |---|      6 = Player2's mancala
                |===============================|    7-12 = Player2's pits
                      12  11  10  9   8   7
        """
        if pits:
            assert len(pits) == len(self._pits_default)
        else:
            pits = list(self._pits_default)

        self._pits = pits
        self._p1 = 1  # p1 == 1 for Player1, Player2 == 0

    @property
    def pits(self) -> list[int]:
        """Return list of all pits, including mancalas"""
        return list(self._pits)

    @property
    def p1_mancala(self) -> int:
        """Return the number of seeds in P1's mancala"""
        return self.pits[self._mancala_idxs[1]]

    @property
    def p2_mancala(self) -> int:
        """Return the number of seeds in P2's mancala"""
        return self.pits[self._mancala_idxs[0]]

    @property
    def p1_pits(self) -> list[int]:
        """Return list containing the number of seeds in each of P1's pits """
        return self._pits[0:6]

    @property
    def p2_pits(self) -> list[int]:
        """Return list containing the number of seeds in each of P2's pits """
        return self._pits[7:13]

    @property
    def p1(self) -> int:
        """Return 1 if p1's turn, return 0 if p2's turn"""
        return self._p1

    @property
    def game_over(self) -> bool:
        """Game is over when ...
            1. All pits on either side of the board are empty or
            2. Either mancala has more than half of the seeds
        """
        return any([self.side_empty,
                    self.p1_mancala >= 25,
                    self.p2_mancala >= 25])

    @property
    def game_score(self) -> tuple[int, int]:
        """Return the game score as an integer tuple of (p1_score, p2_score)"""
        return (self.p1_mancala, self.p2_mancala)

    @property
    def game_winner(self) -> int:
        """Return an int of the game winner: 1 if P1, 2 if P2"""
        if not self.game_over:
            return 0
        return 1 if self.game_score[0] > self.game_score[1] else 2

    @property
    def side_empty(self) -> bool:
        """Return True if all of the pits on either side are empty"""
        return any([sum(self.p1_pits) == 0, sum(self.p2_pits) == 0])

    def render(self) -> str:
        """Render the game board along with the number of seeds and indices of each pit"""

        board = "\n         0   1   2   3   4   5\n   |===============================|\n   "
        board += "|---| {:<2}| {:<2}| {:<2}| {:<2}| {:<2}| {:<2}|---|".format(
            self.p1_pits[0], self.p1_pits[1], self.p1_pits[2],
            self.p1_pits[3], self.p1_pits[4], self.p1_pits[5])
        board += f"\n13 |{self.p1_mancala:>2} |=======================| {self.p2_mancala:<2}| 6"
        board += "\n   |---| {:<2}| {:<2}| {:<2}| {:<2}| {:<2}| {:<2}|---|\n   ".format(
            self.p2_pits[5], self.p2_pits[4], self.p2_pits[3],
            self.p2_pits[2], self.p2_pits[1], self.p2_pits[0])
        board += "|===============================|\n         12  11  10  9   8   7\n"

        return board

    def sow(self, pit: int) -> tuple[int, int]:
        """Sow seeds in a counter-clockwise circle

        Players can only grab from their own pits.
        Players cannot sow seeds from their own mancala.
        Players cannot sow seeds into their opponent's mancala.

        Args:
            pit (int): Index of pit. 0-5 for Player1, 7-12 for Player2

        Returns:
            int: Index of pit where the final seed was sowed
        """
        print(f"Player {1 if self.p1 else 2} selected pit {pit}: {self.pits[pit]} seeds")

        illegal_moves = [
            # Can't move if the game is over
            # self.game_over,
            # Can't sow seeds from either Mancala
            pit in self._mancala_idxs,
            # Can't sow seeds if the pit is empty
            self.pits[pit] == 0,
            # P1 cannot sow seeds from P2's pits
            pit in self._p2_range and self.p1,
            # P2 cannot sow seeds from P1's pits
            pit in self._p1_range and not self.p1,
            # Invalid pit index provided
            pit not in self._pit_idxs
        ]

        # TODO: Raise IllegalMoveError
        if any(illegal_moves):
            raise IllegalMoveError

        # Grab seeds from pit and set the pit to be empty
        seeds = self.pits[pit]
        self._pits[pit] = 0
        current_pit = pit

        # Place one seed in each pit in a counter-clockwise fashion until no seeds are left in the
        # player's hand
        while seeds > 0:
            # Counter-clockwise movement
            current_pit = (current_pit - 1) % len(self.pits)

            if self.p1 and current_pit == self._mancala_idxs[0]:
                continue
            # Players cannot sow seeds into eachother's mancala
            if not self.p1 and current_pit == self._mancala_idxs[1]:
                continue
            self._pits[current_pit] += 1
            seeds -= 1

        # If the last seed lands in their own mancala, the player goes again
        if current_pit in self._mancala_idxs and not self.side_empty:
            return self.game_score

        # If the last seed lands in an empty pit on the player's own side, capture the seed in the
        # current pit AND all seeds from the directly opposite pit and place them in the current
        # player's mancala
        if self.pits[current_pit] == 1 and self.pits[12-current_pit] >= 1:
            # Determine if player landed in an empty pit on their own side
            if (self.p1 and current_pit in self._p1_range) or (not self.p1 and current_pit in self._p2_range):
                opposite_pit = self.pits[12 - current_pit]
                self._pits[self._mancala_idxs[self.p1]] += sum([opposite_pit, 1])

                # Set both pits to 0
                self._pits[12 - current_pit] = 0
                self._pits[current_pit] = 0

        # If all pits are empty on either side, the game is over
        # Sum the remaining seeds and add them to the respective mancala
        if self.side_empty:
            self._pits[self._mancala_idxs[1]] += sum(self.p1_pits)
            self._pits[self._mancala_idxs[0]] += sum(self.p2_pits)
            self._pits[self._p1_range.start:self._p1_range.stop] = [0]*6
            self._pits[self._p2_range.start:self._p2_range.stop] = [0]*6
            return self.game_score

        # If the last seed lands in the same pit they selected, the player goes again
        self._p1 = self.p1 if pit == current_pit else not self.p1
        return self.game_score
