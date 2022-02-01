import pygame as pg
from pytmx import load_pygame, TiledTileLayer
from config import TILESIZE, WHITE


class TiledMap:
    def __init__(self, filename):
        tm = load_pygame(filename, pixelalpha=True)
        self.width = int(tm.width * TILESIZE)
        self.height = int(tm.height * TILESIZE)
        self.tmxdata = tm

    def render(self, surface):
        ti = self.tmxdata.get_tile_image_by_gid
        for layer in self.tmxdata.visible_layers:
            if isinstance(layer, TiledTileLayer):
                # ! Not a fan of black's formatting decision here
                for (
                    x,
                    y,
                    gid,
                ) in layer:
                    tile = ti(gid)
                    if tile:
                        tile = pg.transform.scale(tile, (TILESIZE, TILESIZE))
                        surface.blit(
                            tile,
                            (
                                int(x * TILESIZE),
                                int(y * TILESIZE),
                            ),
                        )

    def make_map(self):
        temp_surface = pg.Surface((self.width, self.height))
        self.render(temp_surface)
        return temp_surface


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
        self.font = pg.font.SysFont("comicsansms", 12)
        self.image = self.font.render(self.symbol, True, WHITE)

    def draw(self):
        pass

    def update(self):
        # Agent's visual attributes
        self.image = self.font.render(self.symbol, True, WHITE)
        self.rect = self.image.get_rect()
        self.game.screen.blit(source=self.image, dest=(self.rect.x, self.rect.y))
