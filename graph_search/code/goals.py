import pygame as pg
from config import TILESIZE, GREEN
from random import choice, randrange


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
        self._collision_wall()

    def _collision_wall(self) -> None:
        """Hacky solution to prevent Goals from sticking inside Walls"""
        hits = pg.sprite.spritecollide(sprite=self, group=self.game.walls, dokill=False)
        while hits:
            random_road = choice(self.game.roads.sprites())
            x = random_road.x + (randrange(random_road.rect.width) / 2)
            y = random_road.y + (randrange(random_road.rect.height) / 2)
            self.rect = pg.Rect(x, y, self.rect.width, self.rect.height)
            hits = pg.sprite.spritecollide(
                sprite=self, group=self.game.walls, dokill=False
            )


class Teleport(Goal):
    def __init__(self, game, x: float, y: float):
        super().__init__(game, x, y)
