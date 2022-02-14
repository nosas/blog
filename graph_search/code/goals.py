import pygame as pg
from config import TILESIZE, GREEN


class Goal(pg.sprite.Sprite):
    def __init__(self, game, x: float, y: float):
        self.game = game
        self.groups = (
            game.all_sprites,
            game.goals,
        )
        pg.sprite.Sprite.__init__(self, self.groups)

        self.pos = pg.Vector2(x, y)

        width, height = (TILESIZE, TILESIZE)
        self.image = pg.Surface((width, height))
        self.image.fill(color=GREEN)
        self.rect = pg.Rect(self.pos.x, self.pos.y, width, height)


class Teleport(Goal):
    def __init__(self, game, x: float, y: float):
        super().__init__(game, x, y)
