from abc import abstractmethod
from sys import maxsize

import pygame as pg

from config import BLACK, RED
from helper import calculate_point_dist
from objects import Wall
from typing import List, Tuple


class Sensor(pg.sprite.Sprite):
    """Base class for Agent's Sensors. Draws lines from Agent to some point"""

    def __init__(self, game, agent):
        self.game = game
        self.agent = agent

        self.groups = (self.game.sensors,)
        pg.sprite.Sprite.__init__(self, self.groups)

    @abstractmethod
    def draw(self):
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    def update(self):
        self.draw()


class CardinalSensor(Sensor):
    def __init__(self, game, agent):
        super().__init__(game=game, agent=agent)
        self.line_thickness = 2  # 2px
        self._screen_height = game.screen.get_height()
        self._screen_width = game.screen.get_height()

    @property
    def _collisions(self) -> List[pg.sprite.Sprite]:
        # return pg.sprite.spritecollide(sprite=self, group=self.game.walls, dokill=False)
        groups = [self.game.mobs, self.game.goals, self.game.walls]
        collisions = []
        for group in groups:
            new_collisions = pg.sprite.spritecollide(
                sprite=self, group=group, dokill=False
            )
            if new_collisions:
                collisions += new_collisions
        return collisions

    def _find_nearest_collision(self, direction: str) -> pg.sprite.Sprite:
        """Return an object that is closest to the sprite in some direction NSEW"""
        nearest_obj = None
        if len(self._collisions) == 1:
            nearest_obj = self._collisions[0]
        elif len(self._collisions) > 1:
            nearest_obj_dist = maxsize
            for obj in self._collisions:
                obj_rect_coord = {
                    "N": obj.rect.midbottom,
                    "S": obj.rect.midtop,
                    "E": obj.rect.midleft,
                    "W": obj.rect.midright,
                }
                dist = calculate_point_dist(
                    point1=self.agent.pos, point2=obj_rect_coord[direction]
                )
                if dist < nearest_obj_dist:
                    nearest_obj = obj
                    nearest_obj_dist = dist

        return nearest_obj

    @property
    def _north(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly North of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self.line_thickness, self._screen_height),
        )
        self.rect.bottom = self.agent.hit_rect.top
        nearest_wall = self._find_nearest_collision(direction="N")
        return nearest_wall

    @property
    def _south(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly South of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self.line_thickness, self._screen_height),
        )
        self.rect.top = self.agent.hit_rect.bottom
        nearest_wall = self._find_nearest_collision(direction="S")
        return nearest_wall

    @property
    def _east(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly East of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self._screen_width, self.line_thickness),
        )
        self.rect.left = self.agent.hit_rect.right
        nearest_wall = self._find_nearest_collision(direction="E")
        return nearest_wall

    @property
    def _west(self) -> Wall:
        """Return the object (Wall/Mob/Goal) located directly West of the Agent"""
        self.rect = pg.Rect(
            self.agent.hit_rect.center,
            (self._screen_width, self.line_thickness),
        )
        self.rect.right = self.agent.hit_rect.left
        nearest_wall = self._find_nearest_collision(direction="W")
        return nearest_wall

    def _draw_north_south(self) -> None:
        for wall, color in zip([self._north, self._south], [RED, BLACK]):
            if wall:
                y = wall.rect.bottom if wall == self._north else wall.rect.top
                pg.draw.line(
                    surface=self.game.screen,
                    color=color,
                    start_pos=self.agent.pos,
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
                    start_pos=self.agent.pos,
                    end_pos=(x, self.agent.hit_rect.centery),
                    width=self.line_thickness,
                )

    def draw(self):
        self._draw_north_south()
        self._draw_east_west()

    def update(self):
        return super().update()


class ObjectSensor(Sensor):
    def __init__(
        self, game, agent, group: pg.sprite.Group, color: Tuple[int, int, int]
    ):
        super().__init__(game=game, agent=agent)
        self._color = color
        self._nearest_obj = None
        self._obj_group = group
        self._find_nearest_obj()

    @property
    def nearest(self) -> pg.sprite.Sprite:
        return self._nearest_obj

    @property
    def dist(self) -> pg.sprite.Sprite:
        return (
            calculate_point_dist(point1=self.agent.pos, point2=self.nearest.rect.center)
            if self.nearest
            else None
        )

    def _find_nearest_obj(self) -> None:
        def is_closer(obj, dist: int) -> bool:
            return (
                True
                if self._nearest_obj is None
                else (obj is not self.nearest) and (dist < self.dist)
            )

        def set_nearest_obj(new_obj) -> None:
            if self.nearest is not None:
                self._nearest_obj.is_nearest_obj = False
            self._nearest_obj = new_obj
            self._nearest_obj.is_nearest_obj = True

        for obj in self._obj_group.sprites():
            dist = calculate_point_dist(point1=self.agent.pos, point2=obj.pos)
            if is_closer(obj=obj, dist=dist):
                set_nearest_obj(new_obj=obj)

    def draw(self) -> None:
        self._find_nearest_obj()
        if self.nearest:
            pg.draw.line(
                surface=self.game.screen,
                color=self._color,
                start_pos=self.agent.pos,
                end_pos=self.nearest.rect.center,
                width=2,
            )

    def update(self):
        return super().update()
