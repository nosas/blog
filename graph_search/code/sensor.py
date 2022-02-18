from sys import maxsize

import pygame as pg

from config import BLACK, RED
from helper import calculate_point_dist
from objects import Wall


# TODO Refactor Sensor to be an abstract base class
# TODO Create MobSensor
class Sensor(pg.sprite.Sprite):
    """Draw lines from Agent to some point"""

    def __init__(self, game, agent):
        self.game = game
        self.agent = agent

        self.groups = (self.game.sensors,)
        pg.sprite.Sprite.__init__(self, self.groups)

    def draw(self):
        pass

    def update(self):
        self.draw()


class CardinalSensor(Sensor):
    def __init__(self, game, agent):
        super().__init__(game, agent)
        self.line_thickness = 2  # 2px
        self._screen_height = game.screen.get_height()
        self._screen_width = game.screen.get_height()

    # TODO Refactor to be more generalize, find_nearest_object. Replace collisions with
    # TODO list of groups to include, use spritecollideany and filter.
    @staticmethod
    def _find_nearest_wall(
        sprite: pg.sprite.Sprite, collisions: list[pg.sprite.Sprite], direction: str
    ) -> Wall:
        nearest_wall = None

        if len(collisions) == 1:
            nearest_wall = collisions[0]
        elif len(collisions) > 1:
            nearest_dist = maxsize
            for wall in collisions:
                wall_coord = {
                    "N": wall.rect.midbottom,
                    "S": wall.rect.midtop,
                    "E": wall.rect.midleft,
                    "W": wall.rect.midright,
                }
                dist = calculate_point_dist(
                    point1=sprite.hit_rect.center, point2=wall_coord[direction]
                )
                if dist < nearest_dist:
                    nearest_wall = wall
                    nearest_dist = dist

        return nearest_wall

    @property
    def _collisions(self) -> list[pg.sprite.Sprite]:
        return pg.sprite.spritecollide(sprite=self, group=self.game.walls, dokill=False)

    @property
    def _north(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly North of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self.line_thickness, self._screen_height),
        )
        self.rect.bottom = self.agent.hit_rect.top
        nearest_wall = CardinalSensor._find_nearest_wall(
            sprite=self.agent, collisions=self._collisions, direction="N"
        )
        return nearest_wall

    @property
    def _south(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly South of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self.line_thickness, self._screen_height),
        )
        self.rect.top = self.agent.hit_rect.bottom
        nearest_wall = CardinalSensor._find_nearest_wall(
            sprite=self.agent, collisions=self._collisions, direction="S"
        )
        return nearest_wall

    @property
    def _east(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly East of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self._screen_width, self.line_thickness),
        )
        self.rect.left = self.agent.hit_rect.right
        nearest_wall = CardinalSensor._find_nearest_wall(
            sprite=self.agent, collisions=self._collisions, direction="E"
        )
        return nearest_wall

    @property
    def _west(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly West of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self._screen_width, self.line_thickness),
        )
        self.rect.right = self.agent.hit_rect.left
        nearest_wall = CardinalSensor._find_nearest_wall(
            sprite=self.agent, collisions=self._collisions, direction="W"
        )
        return nearest_wall

    def _draw_north_south(self) -> None:
        for wall, color in zip([self._north, self._south], [RED, BLACK]):
            if wall:
                y = wall.rect.bottom if wall == self._north else wall.rect.top
                pg.draw.line(
                    surface=self.game.screen,
                    color=color,
                    start_pos=self.agent.hit_rect.center,
                    end_pos=(self.agent.hit_rect.centerx, y),
                    width=self.line_thickness,
                )

    def _draw_east_west(self) -> None:
        for wall in [self._east, self._west]:
            if wall:
                x = wall.rect.left if wall == self._east else wall.rect.right
                pg.draw.line(
                    surface=self.game.screen,
                    color=BLACK,
                    start_pos=self.agent.hit_rect.center,
                    end_pos=(x, self.agent.hit_rect.centery),
                    width=self.line_thickness,
                )

    def draw(self):
        self._draw_north_south()
        self._draw_east_west()
