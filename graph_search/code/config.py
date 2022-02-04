from pygame import Rect

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Game window settings
TILESIZE = 16
WIDTH = 100 * TILESIZE
HEIGHT = 40 * TILESIZE

# Game settings
DEBUG = False
FPS = 100
TITLE = "Agent Movement Demo"
BG_COLOR = BLACK

# Map settings
MAP = "map16.tmx"
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

# Agent settings
AGENT_IMG = "agent.png"
AGENT_SPEED = 0.026 * 1.5
AGENT_ROT_SPEED = 0.12
AGENT_HIT_RECT = Rect(0, 0, TILESIZE * 0.7, TILESIZE * 0.7)
AGENT_RANDOM_SPAWN = False

# Mob settings
NUM_MOBS = 100
MOB_SIZE = (TILESIZE * 0.7, TILESIZE * 0.7)
MOB_SPEED = AGENT_SPEED * 0.7
