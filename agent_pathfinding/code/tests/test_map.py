import pygame as pg

from game import Game


class TestMap:
    g = Game(manual=True)
    g.new()

    def test_map_is_a_tile(self):
        # agent_spawn = pg.Vector2((self.g.agent.pos.y, self.g.agent.pos.x))
        assert self.g.map.is_a_tile(tx=self.g.agent.tpos.x, ty=self.g.agent.tpos.y)

    def test_map_is_a_tile_fail(self):
        # agent_spawn = self.g.agent.pos
        empty_top_left = pg.Vector2((0, 0))
        assert not self.g.map.is_a_tile(tx=empty_top_left.x, ty=empty_top_left.y)

    def test_map_is_a_tile_out_of_bounds_fail(self):
        """Verify a position (x,y) is not out of the map's bounds"""
        out_of_bounds = [(-100, -40), (-1, -1), (-1, 1), (1, -1), (100, 40), (400, 500)]
        for tx, ty in out_of_bounds:
            assert not self.g.map.is_a_tile(tx=tx, ty=ty)


class TestMapTrain:
    g = Game(
        manual=True,
        map_name="map_train1.tmx",
        rand_agent_spawn=True,
        rand_goal_spawn=True,
    )

    def test_map_train_creation(self):
        self.g.new()
        assert self.g.agent.nearest_goal is not None
