# Minimax search in Mancala

Minimax search is a fundamental depth-first search algorithm used in Artificial Intelligence, decision theory, game theory, and statistics.
The purpose of minimax search is to find  **minimize the maximimum loss** in a worst case scenario.

This algorithm can be implemented in *n*-player games but is most commonly implemented in turn-based 2-player games such as: checkers, chess, connect 4, mancala, etc.
We'll focus on 2-player games - specifically mancala - in this article for the sake of simplicity.

---
## Minimax vocabulary

Let's familiarize ourselves with minimax-related variables and keywords before we implement minimax search.

| **Keyword**      | **Variable**   | **Definition** |
| --------------   | -------------- | -------------- |
| Player           | p<sub>max</sub>| Maximizes their action's value and minimizes the maximum loss in a worst case scenario. |
 Opponent          | p<sub>min</sub>| Minimizes the value of the player's (p<sub>max</sub>) action |
| Heuristic value  | `v`            | Value of the player's action obtained by an evaluation function. Larger numbers are more beneficial for p<sub>max</sub>, whereas smaller numbers are more benficial for p<sub>min</sub>. |
| Branching Factor | `b`            | How many actions the player has available |
| Depth            | `d`            | How deep into the tree - or how many moves ahead - the algorithm will search |

---
## Minimax example

Suppose we're playing a 2-player turn-based game where each player has a choice between two actions per turn - so our branching factor, `b`, will be equal to 2.

