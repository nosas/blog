import pygame as pg

from helper import calculate_point_dist


class TestHelper:
    def test_calculate_point_dist(self):
        p1 = pg.Vector2(0, 0)
        p2 = pg.Vector2(1, 1)
        dist = calculate_point_dist(p1, p2)
        assert dist.round(2) == 1.41

        p2 = pg.Vector2(-1, -1)
        dist = calculate_point_dist(p1, p2)
        assert dist.round(2) == 1.41
