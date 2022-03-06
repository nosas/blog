import pygame as pg

from config import TILESIZE
from game import Game


class TestMap:
    g = Game(manual=True)
    g.new()

    def test_map_is_a_tile(self):
        # agent_spawn = pg.Vector2((self.g.agent.pos.y, self.g.agent.pos.x))
        assert self.g.map.is_a_tile(self.g.agent.pos)

    def test_map_is_a_tile_fail(self):
        # agent_spawn = self.g.agent.pos
        empty_top_left = pg.Vector2((0, 0))
        assert not self.g.map.is_a_tile(empty_top_left)

    def test_map_is_a_tile_out_of_bounds_fail(self):
        """Verify a position (x,y) is not out of the map's bounds"""
        out_of_bounds = [
            (-100, -40),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (100, 40),
            (400, 500)
        ]

        for x, y in out_of_bounds:
            assert not self.g.map.is_a_tile(pg.Vector2(x, y) * TILESIZE)
