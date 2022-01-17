from agent import AgentManual, AgentRandom
from mancala import Mancala
from mancala_errors import IllegalMoveError


if __name__ == "__main__":
    b = Mancala()
    print(b.render())

    agents = {
        0: AgentRandom(),  # Player 2
        1: AgentManual()   # Player 1
    }

    while not b.game_over:
        agent = agents[b.p1]
        possible_moves = agent.get_possible_moves(game=b)

        print(f"Player {1 if b.p1 else 2}'s Turn! Choose a number in range {possible_moves}")
        pit = agent.move(game=b)

        # Stop the game is player inputs 'q'
        if pit == 'q':
            break

        try:
            b.sow(pit=int(pit))
            print(b.render())
        except IllegalMoveError:
            print("[!] Illegal move!")
        except ValueError:
            print("[!] Invalid input, must be in ['q', range(14)]. Try again")

    print(f"Winner! Player{b.game_winner} with total score {b.game_score}")
