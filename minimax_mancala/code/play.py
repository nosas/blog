from agent import Agent, AgentManual, AgentMinimaxAlphaBeta
from mancala import Mancala
from mancala_errors import IllegalMoveError


def play_mancala(board: Mancala, agents: dict[int, Agent]) -> None:
    print(board.render())
    move_history = []

    while not board.game_over:
        agent = agents[board.p1]
        possible_moves = agent.get_possible_moves(game=board)

        print(f"Player {1 if board.p1 else 2}'s Turn! Choose a number in range {possible_moves}")
        pit = agent.move(game=board)
        # Stop the game if player inputs 'q'
        if pit in ['q', 'Q']:
            break

        try:
            current_player = 1 if board.p1 else 2
            pit = int(pit)
            if pit not in possible_moves:
                raise ValueError

            seeds = board.pits[pit]
            score_diff, capture, go_again = board.sow(pit=pit)
            move_history.append(pit)

            print(f"    Player {current_player} selected pit {pit}: "
                  f"{seeds} seeds (+{score_diff}, {capture:>}, {go_again:>})")
            print(board.render())
        except IllegalMoveError:
            print("[!] Illegal move!")
        except ValueError:
            print(f"[!] Invalid input, must be in ['q', {possible_moves}]. Try again")

    print()
    print(f"Winner! Player{board.game_winner}")
    print(f"Player scores: {board.game_score[1], board.game_score[0]}")
    print(f"Player moves : {move_history}")


if __name__ == "__main__":
    board = Mancala()
    agents = {
        board.p1: AgentMinimaxAlphaBeta(depth=7, p1=board.p1),    # Player 1
        not board.p1: AgentManual()                               # Player 2
    }

    play_mancala(board=board, agents=agents)
