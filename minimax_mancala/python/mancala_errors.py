class IllegalMoveError(Exception):
    """Raised when an illegal move is selected

    Illegal moves include:
        1. Moving when the game is over
        2. Sowing seeds from an empty pit
        3. Sowing seeds from either mancala
        4. P1 sows seeds from P2's pits
        5. P2 sows seeds from P1's pits
        6. Invalid pit index provided, must be in range (0-13)
    """
    pass


