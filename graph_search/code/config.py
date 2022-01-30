from pygame import Rect

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Game window settings
WIDTH = 1024 * 2
HEIGHT = 900

# Game settings
FPS = 100
TITLE = "Agent Movement Demo"
BG_COLOR = BLACK

# Map settings
TILESIZE = 32
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

# Agent settings
AGENT_IMG = "agent.png"
AGENT_SPEED = 0.5
AGENT_ROT_SPEED = 0.25
AGENT_HIT_RECT = Rect(0, 0, 32, 32)
