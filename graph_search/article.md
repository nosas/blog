<title>Agent Movement with Graph search and Reinforced Learning</title>

# Agent Movement

I want to learn how to make an agent - some automated object, such as a car or NPC - move from point A to point B.
It should be able to dodge collisions and reroute on the fly.

---
## Problem

Video games contain many repetitive, time-consuming tasks.
Repetitive tasks include fetching items back and forth from NPCs, item seeking and gathering, or defeating X number of opponents.


---
## Solution

An agent capable of moving from point A to B with graph search and reinforced learning.
More specifically: dodge obstacles and complete fetching, gathering, and defeating tasks.

Its movement and pathfinding will not be most optimal *just* yet.
We'll explore and document enhancements in future articles.

---
## Solution Framework

We'll be using Python3 for this whole project:

* PyGame to create the game and simulate tasks
* Numpy for the graph search (?)
* Pandas to visualize the data

I'll also be testing [Black](https://pypi.org/project/black/)

See `requirement.txt` for the packages and package versions used for this article.

