import pygame as pg
from config import WHITE


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
    def __init__(self, game, x: int, y: int, width: int, height: int, groups: tuple[pg.sprite.Group] = None):
        self.game = game
        if groups is not None:
            self.groups = groups + (game.all_sprites, game.roads)
        else:
            self.groups = game.all_sprites, game.roads
        pg.sprite.Sprite.__init__(self, self.groups)

        self.rect = pg.Rect(x, y, width, height)
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y


class Path(Road):
    _font = {"name": "comicsansms", "size": 12}
    _symbols = {"left": "L", "right": "R", "up": "U", "down": "D"}

    @staticmethod
    def _get_symbol(direction: str) -> str:
        return Path._symbols[direction]

    @property
    def _symbol(self) -> str:
        return Path._get_symbol(direction=self.direction)

    def __init__(self, game, x: int, y: int, width: int, height: int, direction: str):
        super().__init__(game=game, x=x, y=y, width=width, height=height, groups=(game.paths,))

        self.direction = direction
        self.font = pg.font.SysFont(**Path._font)
        self.image = self.font.render(self._symbol, True, WHITE)

    def draw(self):
        self.image = self.font.render(self._symbol, True, WHITE)
        # self.rect = self.image.get_rect()  # ! Bug! Font renders on screen's top-left
        self.game.screen.blit(source=self.image, dest=(self.x, self.y))

    def update(self):
        pass


class Sidewalk(Road):
    def __init__(self, game, x: int, y: int, width: int, height: int):
        super().__init__(game=game, x=x, y=y, width=width, height=height, groups=(game.sidewalks,))
        self.image = pg.Surface(size=(self.rect.width, self.rect.height))

    def update(self):
        pass
