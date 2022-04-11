import numpy as np
import pygame as pg
from pytmx import TiledTileLayer, load_pygame

from config import TILESIZE
from random import choice


class TiledMap:
    def __init__(self, filename: str):
        tm = load_pygame(filename=filename, pixelalpha=True)
        self.height = int(tm.height * TILESIZE)
        self.width = int(tm.width * TILESIZE)
        self.theight = tm.height
        self.twidth = tm.width

        self.tmxdata = tm
        # Map properties for displaying game/obs/train info boxes
        self.properties = {
            k: np.array(v.split(",")).astype("int") * TILESIZE
            for k, v in self.tmxdata.properties.items()
        }

        # True if tile is part of the Map, False if outside the Map
        self._binary = np.zeros(shape=(self.theight, self.twidth), dtype=np.bool8)
        # True if tile is a Wall
        self._walls = np.zeros(shape=(self.theight, self.twidth), dtype=np.bool8)
        # True if tile is part of the Map and not a Wall
        self._walkable = np.zeros(shape=(self.theight, self.twidth), dtype=np.bool8)

    @property
    def binary(self) -> np.array:
        """Return a numpy array representation of the Maps's tiles"""
        return self._binary

    @property
    def walls(self) -> np.array:
        """Return a numpy array representation of the Maps's Walls"""
        return self._walls

    @property
    def walkable(self) -> np.array:
        """Return a numpy array representation of the Maps's walkable tiles"""
        return self._walkable

    def get_random_tile(self) -> pg.Vector2:
        """Return tile position of a random tile"""
        ty, tx = choice(np.argwhere(np.array(self.binary) == 1))
        return pg.Vector2(tx, ty)

    def get_tile_type(self, tx: int, ty: int) -> str:
        """Return the name of a tile based on the tile's layer name"""
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

    def is_a_wall(self, tx: int, ty: int) -> bool:
        tx, ty = (int(tx), int(ty))
        return self.is_a_tile(tx=tx, ty=ty) and self.walls[ty][tx]

    def is_walkable(self, tx: int, ty: int) -> bool:
        tx, ty = (int(tx), int(ty))
        return self.is_a_tile(tx=tx, ty=ty) and self.walkable[ty][tx]

    def render(self, surface: pg.Surface) -> None:
        def map_layer_to_binary(layer_name: str, tx: int, ty: int) -> None:
            """Build binary maps for each layer"""
            self._binary[ty][tx] = True
            if layer_name in ["wall"]:
                self._walls[ty][tx] = True
            elif layer_name in ["sidewalk", "road", "street"]:
                self._walkable[ty][tx] = True

        ti = self.tmxdata.get_tile_image_by_gid
        for layer in self.tmxdata.visible_layers:
            if isinstance(layer, TiledTileLayer):
                for (tx, ty, gid) in layer:
                    tile = ti(gid)
                    if tile:
                        map_layer_to_binary(layer_name=layer.name, tx=tx, ty=ty)
                        tile = pg.transform.scale(tile, (TILESIZE, TILESIZE))
                        surface.blit(
                            tile,
                            (
                                int(tx * TILESIZE),
                                int(ty * TILESIZE),
                            ),
                        )

    def make_map(self) -> pg.Surface:
        """Create Surface and render images/tiles/sprites/etc."""
        temp_surface = pg.Surface((self.width, self.height))
        self.render(temp_surface)
        return temp_surface
