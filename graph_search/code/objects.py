import pygame as pg


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
