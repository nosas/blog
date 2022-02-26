import pygame as pg
from pytmx import TiledTileLayer, load_pygame

from config import TILESIZE


class TiledMap:
    def __init__(self, filename: str):
        tm = load_pygame(filename=filename, pixelalpha=True)
        self.width = int(tm.width * TILESIZE)
        self.height = int(tm.height * TILESIZE)
        self.tmxdata = tm

    def render(self, surface: pg.Surface) -> None:
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

    def make_map(self) -> pg.Surface:
        """Create Surface and render images/tiles/sprites/etc."""
        temp_surface = pg.Surface((self.width, self.height))
        self.render(temp_surface)
        return temp_surface