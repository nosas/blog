import numpy as np
import pygame as pg
from pytmx import TiledTileLayer, load_pygame

from config import TILESIZE
from random import choice


class TiledMap:
    def __init__(self, filename: str):
        tm = load_pygame(filename=filename, pixelalpha=True)
        self.width = int(tm.width * TILESIZE)
        self.height = int(tm.height * TILESIZE)
        self.tmxdata = tm
        # Map properties for displaying game/obs/train info boxes
        self.properties = {
            k: np.array(v.split(",")).astype("int") * TILESIZE
            for k, v in self.tmxdata.properties.items()
        }

        self._binary = np.zeros(shape=(tm.height, tm.width), dtype=np.bool8)

    @property
    def binary(self) -> np.array:
        """Return a numpy array representation of the map's tiles"""
        return self._binary

    def get_random_tile(self) -> pg.Vector2:
        """Return tile position of a random tile"""
        ty, tx = choice(np.argwhere(np.array(self.binary) == 1))
        return pg.Vector2(tx, ty)

    def get_tile_type(self, tx: int, ty: int) -> str:
        tx = int(tx)
        ty = int(ty)

        if not self.is_a_tile(tx=tx, ty=ty):
            return None

        for lid in self.tmxdata.visible_tile_layers:
            lname = self.tmxdata.layers[lid].name
            ldata = np.asarray(self.tmxdata.layers[lid].data)
            try:
                if ldata[ty][tx] != 0:
                    return lname
            except ValueError:
                pass

    def is_a_tile(self, tx: int, ty: int) -> bool:
        tx, ty = (int(tx), int(ty))

        checks = [
            0 > tx or tx >= self.binary.shape[1],
            0 > ty or ty >= self.binary.shape[0],
        ]
        if any(checks):
            return False
        return self.binary[ty][tx]

    def render(self, surface: pg.Surface) -> None:
        ti = self.tmxdata.get_tile_image_by_gid
        for layer in self.tmxdata.visible_layers:
            if isinstance(layer, TiledTileLayer):
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
