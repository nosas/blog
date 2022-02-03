import pygame as pg
from config import TILESIZE, WHITE


class Wall(pg.sprite.Sprite):
    def __init__(self, game, x: int, y: int, width: int, height: int):
        self.groups = game.walls
        pg.sprite.Sprite.__init__(self, self.groups)

        self.game = game
        self.rect = pg.Rect(x, y, width, height)
        self.hit_rect = self.rect
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y


class Road(pg.sprite.Sprite):
    def __init__(self, game, x: int, y: int):
        self.game = game
        self.groups = game.all_sprites, game.roads
        pg.sprite.Sprite.__init__(self, self.groups)
        self.rect = pg.Rect(x, y, TILESIZE, TILESIZE)
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Path(pg.sprite.Sprite):
    _font = {"name": "comicsansms", "size": 12}
    _symbols = {"left": "L", "right": "R", "up": "U", "down": "D"}

    @staticmethod
    def _get_symbol(direction: str) -> str:
        return Path._symbols[direction]

    @property
    def symbol(self) -> str:
        return Path._get_symbol(direction=self.direction)

    def __init__(self, game, x: int, y: int, width: int, height: int, direction: str):
        self.game = game
        self.groups = game.roads
        pg.sprite.Sprite.__init__(self, self.groups)

        self.rect = pg.Rect(x, y, width, height)
        self.x = x
        self.y = y
        self.rect.x = x * width
        self.rect.y = y * height

        self.direction = direction
        self.font = pg.font.SysFont(**Path._font)
        self.image = self.font.render(self.symbol, True, WHITE)

    def draw(self):
        self.image = self.font.render(self.symbol, True, WHITE)
        # self.rect = self.image.get_rect()  # ! Bug! Font renders on screen's top-left
        self.game.screen.blit(source=self.image, dest=(self.x, self.y))

    def update(self):
        pass
