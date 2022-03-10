import pygame as pg

from config import GREEN, TILESIZE


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

    @property
    def tpos(self) -> pg.Vector2:
        return self.pos / TILESIZE

    def _collision_wall(self) -> None:
        """Hacky solution to prevent Goals from sticking inside Walls"""
        hits = pg.sprite.spritecollide(sprite=self, group=self.game.walls, dokill=False)
        while hits:
            random_tpos = self.game.map.get_random_tile()
            x = random_tpos.x * TILESIZE
            y = random_tpos.y * TILESIZE
            self.rect = pg.Rect(x, y, self.rect.width, self.rect.height)
            hits = pg.sprite.spritecollide(
                sprite=self, group=self.game.walls, dokill=False
            )


class Teleport(Goal):
    def __init__(self, game, x: float, y: float):
        super().__init__(game, x, y)
