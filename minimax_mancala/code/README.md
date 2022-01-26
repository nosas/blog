# Mancala

Mancala is a 2-player turn-based board game.
Each player has 6 pits with 4 seeds/stones along with 1 mancala (store) at the end of the board.

## Gameplay

Players take turns picking up all the seeds from one of their 6 pits and placing them one-by-one until they're holding no seeds.
Stones are placed counterclock-wise into pits and in the player's own mancala at the end of the board.
Players must not place seeds in their opponent's mancala.

### Re-turn

There are two exceptions for when a player can go again, or **re-turn**:

1. The last stone in the player's hand lands in their own mancala
1. The last stone in the player's hand lands in the same pit it started from

### Capture Rule

Lastly, there is a **capture rule**:
If the last stone in the player's hand lands in an empty pit on their own side of the board, and the adjacent pit on the opponent's side contains 1+ seeds, the player may capture all seeds from both pits and place them in their own mancala.

## End game

The player's goal is to have more seeds in their mancala than their opponent.

The game ends on either of two conditions:

1. A player's mancala contains 25+ seeds
1. All pits on a player's side are empty. In this case, the player with seeds still in play may gather and deposit the remaining seeds into their own mancala.

Please watch [this 3-minute video](https://www.youtube.com/watch?v=OX7rj93m6o8) if the explanation above wasn't clear.

---
## Mancala simulator

An example of the simluator can be seen below.
All gameplay occurs on the CLI.

The top-most row is the indices of Player 1's pits.
The following row is Player 1's pits initialized with 4 seeds.
The left-most number is Player 1's mancala index, 13.
The number immediately to the right of Player 1's mancala is his/her score.
Everything else is Player 2's indices, pits, or mancala.

<center>

```

   0   1   2   3   4   5
   |===============================|
   |---| 4 | 4 | 4 | 4 | 4 | 4 |---|
  13 | 0 |=======================| 0 | 6
   |---| 4 | 4 | 4 | 4 | 4 | 4 |---|
   |===============================|
   12  11  10  9   8   7

Player 1's Turn! Choose a number in range [0, 1, 2, 3, 4, 5]
    Player 1 selected pit 1: 4 seeds (+1, 0, 0)

   0   1   2   3   4   5
   |===============================|
   |---| 5 | 0 | 4 | 4 | 4 | 4 |---|
  13 | 1 |=======================| 0 | 6
   |---| 5 | 5 | 4 | 4 | 4 | 4 |---|
   |===============================|
   12  11  10  9   8   7

```

</center>

Each player takes turns selecting a pit.
The simulator will then output which pit the player selected along with four additional outputs:

1. How many seeds were in the pit
1. The player's score difference after playing the move
1. A boolean value of whether the player captured a pit
1. A boolean value of whether the player can go again (re-turn)


## Mancala Agents

In addition to the game simulator, there exists Agent classes inside of `agent.py`.
The agents implement the [Strategy design pattern](https://en.wikipedia.org/wiki/Strategy_pattern), allowing us to create agents with unique Mancala strategies at runtime.
So far, the agents' strategies include: random, minimax, and minimax with alpha-beta pruning.

Additional strategies could be also implemented - such as maximize re-turns, prioritize captures, prevent captures - but we're focusing on minimax for now.

## How to play

Play in the terminal with `python3 play.py`

The minimax agent's depth is currently set to 7 moves ahead, but it can be changed on `line 47` of `play.py`.
I recommend setting the depth between 2-8 moves ahead; depths greater than 8 take too long to choose a move.
The deeper the lookahead, the stronger the Agent.