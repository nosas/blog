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
TILESIZE = 16
WIDTH = 100 * TILESIZE
HEIGHT = 40 * TILESIZE

# Game settings
DEBUG = True
FPS = 100
TITLE = "Agent Movement Demo"
BG_COLOR = BLACK

# Map settings
MAP = "map16.tmx"
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

# Agent settings
AGENT_IMG = "agent.png"
AGENT_SPEED = 0.03 * 1.5
AGENT_ROT_SPEED = 0.12
AGENT_HIT_RECT = Rect(0, 0, 16, 16)
