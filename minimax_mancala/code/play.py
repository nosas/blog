from agent import AgentManual, AgentMinimax
from mancala import Mancala
from mancala_errors import IllegalMoveError

if __name__ == "__main__":
    b = Mancala()
    move_history = []
    print(b.render())

    agents = {
        b.p1: AgentMinimax(depth=6, p1=b.p1),    # Player 1
        not b.p1: AgentManual()                  # Player 2
    }

    while not b.game_over:
        agent = agents[b.p1]
        possible_moves = agent.get_possible_moves(game=b)

        print(f"Player {1 if b.p1 else 2}'s Turn! Choose a number in range {possible_moves}")
        pit = agent.move(game=b)
        # Stop the game if player inputs 'q'
        if pit == 'q':
            break

        try:
            current_player = 1 if b.p1 else 2
            pit = int(pit)
            if pit not in possible_moves:
                raise ValueError

            seeds = b.pits[pit]
            score_diff, capture, go_again = b.sow(pit=pit)
            move_history.append(pit)

            print(f"    Player {current_player} selected pit {pit}: "
                  f"{seeds} seeds (+{score_diff}, {capture:>}, {go_again:>})")
            print(b.render())
        except IllegalMoveError:
            print("[!] Illegal move!")
        except ValueError:
            print(f"[!] Invalid input, must be in ['q', {possible_moves}]. Try again")

    print()
    print(f"Winner! Player{b.game_winner} with total score {b.game_score[1], b.game_score[0]}")
    print(f"Players moves: {move_history}")
