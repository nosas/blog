<title>Agent Pathfinding Part 1: Creating the PyGame environment</title>

# Agent Pathfinding Part 1: Creating the PyGame environment

I want to teach an Agent - some automated object, such as a car or NPC - how to move from point A to point B using reinforcement learning (RL).
The Agent should be able to avoid obstacles, dodge collisions, and reroute as needed on the fly in order to reach the Goal in the most efficient pathing possible.

This post will explain the

---
## Problem

Video games contain many repetitive, time-consuming tasks.
Repetitive tasks include fetching items back and forth from NPCs, item seeking and gathering, or defeating X number of opponents.

These busy tasks - while, admittedly, can be rewarding - are often a time sink in an otherwise exciting game.
I'd like to teach an Agent how to efficiently complete these tasks so I can learn a thing or two from its techniques.

Before we proceed, however, I must admit that I enjoy the repetitive tasks.
Fishing in WoW, chopping trees in Lost Ark, or battling Cogs in ToonTown while listening to a podcast or audiobook is my simple way of winding down.


---
## Solution

An Agent capable of autonomously moving from point A to B while maximizing efficient pathing.
More specifically, the Agent should dodge obstacles and complete fetching, gathering, and battle tasks while pathing at near-maximum efficiency.

---
### Solution Framework

We'll be using Python3.7.9 for this whole project:

- Part 1: Create the game environment with PyGame
- Part 2: Train the Agent with stable-baselines3 and OpenAI Gym
- Part 3: Visualize Agent movement data with pandas and seaborn

I'll also be testing [Black](https://pypi.org/project/black/)

See `requirements.txt` for the packages and package versions used for this article.

---

---
# Pathfinding Game

I created a 2D top-down, tiled-based game using PyGame to simulate Agent pathfinding and collision avoidance.
It looks like this (insert gif).

---
## PyGame

PyGame was chosen over its competitors Unity, GameMaker, and godot because of its plug-and-play capabilities with reinforcement learning frameworks OpenAI Gym and stable-baselines3.
Overall, I'd like to spend more time learning about reinforcement learning while using a language I'm comfortable with.

---
## Map Creation with Tiled and PyTMX

---
### Map Assets

---

---
# The Agent

---
## Agent Movement

The Agent uses `WASD` to move:

- `W` to move forward
- `A` to turn left (or increase heading angle)
- `S` to move backward
- `D` to turn right (or decrease heading angle)

---
### Agent Heading

The Agent's heading angle ranges from 0 to 359 degrees, where 0 degrees is East and 90 degrees is North.

---
## Agent Sensors

The Agent has a handful of sensors to help it "see" - or, make observations - during training:

- **Cardinal Sensor**: Calculates the distance and type of objects located directly North, South, East, and West of the Agent.
- **Object Sensor**: Locates, calculates distance, and draws a line to the nearest Goal or Mob object.
- **Vector Field**: Updated Game's vector field with a 5x5 square to indicate dangers or obstacles in the Agent's immediate surroundings.

---
## Agent Observations

Agent observations are Game facts from the Agent's perspective at each frame of gameplay.
For instance, `dist_to_goal` or `is_hitting_wall` or `heading` are all Game observations specific to the Agent's perspective.

How far away is the Agent from the Goal?
Is the Agent hitting a wall?
What's the Agent's current heading?

---

---
# Goals

When a Goal is spawned, the Map generates a goal-based vector field.

---
## Teleporter Goal

---

---
# Mobs

Mobs are the Agents' opponents.
Upon collision, an Agent and Mob will engage in Battle.
We'll ignore the Battle details for now and consider Battle engagement as a terminating event - aka the Agent loses if it engages in Battle.

Later on, we'll allow the Agent to set the Mob as a Goal.

---
## Mob Types

There are four types of Mobs, each with their own unique color:

- <font style="color: red">A</font>
- <font style="color: cyan">B</font>
- <font style="color: purple">C</font>
- <font style="color: orange">D</font>

There are no other differences between Mob types.
It's simply their color.

The Agent may have tasks where they must defeat X amount of one type of Mob.

---
## Mob Movement

Mobs move along the direction of the map's Path objects.
Path objects are connected together in order to create a loop around the map.
The loop ensures Mob movement is predictable and prevents Mobs from leaving the map.

---

---
# Next Steps

In the next post, we'll train the Agent to autonomously find and reach the Goal.