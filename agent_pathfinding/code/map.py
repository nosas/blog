import numpy as np
import pygame as pg
from pytmx import TiledTileLayer, load_pygame

from config import TILESIZE


class TiledMap:
    def __init__(self, filename: str):
        tm = load_pygame(filename=filename, pixelalpha=True)
        self.width = int(tm.width * TILESIZE)
        self.height = int(tm.height * TILESIZE)
        self._binary = np.zeros(shape=(tm.height, tm.width), dtype=np.bool8)
        self.tmxdata = tm

    @property
    def binary(self) -> np.array:
        """Return a numpy array representation of the map's tiles"""
        return self._binary

    def is_a_tile(self, pos: pg.Vector2) -> bool:
        # TODO Check if pos is out of binary map's bounds/shape
        x = int(pos.x / TILESIZE)
        y = int(pos.y / TILESIZE)

        checks = [
            0 > x or x >= self.binary.shape[1],
            0 > y or y >= self.binary.shape[0],
        ]
        if any(checks):
            return False
        return self.binary[y][x]

    def render(self, surface: pg.Surface) -> None:
        ti = self.tmxdata.get_tile_image_by_gid
        for layer in self.tmxdata.visible_layers:
            if isinstance(layer, TiledTileLayer):
                # ! Not a fan of black's formatting decision here
                for (x, y, gid) in layer:
                    tile = ti(gid)
                    if tile:
                        self._binary[y][x] = True
                        tile = pg.transform.scale(tile, (TILESIZE, TILESIZE))
                        surface.blit(
                            tile,
                            (
                                int(x * TILESIZE),
                                int(y * TILESIZE),
                            ),
                        )

    def make_map(self) -> pg.Surface:
        """Create Surface and render images/tiles/sprites/etc."""
        temp_surface = pg.Surface((self.width, self.height))
        self.render(temp_surface)
        return temp_surface
