from abc import abstractmethod
from sys import maxsize

import pygame as pg

from config import BLACK, RED
from helper import calculate_point_dist
from objects import Wall


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
    def _collisions(self) -> list[pg.sprite.Sprite]:
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


class GoalSensor(Sensor):
    def __init__(self, game, agent):
        super().__init__(game=game, agent=agent)
        self._nearest_goal = None

    @property
    def nearest_goal(self) -> pg.sprite.Sprite:
        return self._nearest_goal

    @property
    def nearest_goal_dist(self) -> pg.sprite.Sprite:
        return (
            calculate_point_dist(point1=self.agent.pos, point2=self.nearest_goal.pos)
            if self.nearest_goal
            else None
        )

    def _find_nearest_goal(self) -> None:
        def is_closer(goal, dist: int) -> bool:
            return (goal is not self.nearest_goal) and (dist < self.nearest_goal_dist)

        def set_nearest_goal(new_goal) -> None:
            if self.nearest_goal is not None:
                self._nearest_goal.is_nearest_goal = False
            self._nearest_goal = new_goal
            self._nearest_goal.is_nearest_goal = True

        for goal in self.game.goals.sprites():
            dist = calculate_point_dist(point1=self.agent.pos, point2=goal.pos)
            if self.nearest_goal is None or is_closer(goal=goal, dist=dist):
                set_nearest_goal(new_goal=goal)

    def draw(self) -> None:
        self._find_nearest_goal()
        if self.nearest_goal:
            pg.draw.line(
                surface=self.game.screen,
                color=BLACK,
                start_pos=self.agent.pos,
                end_pos=self.nearest_goal.rect.center,
                width=2,
            )


class MobSensor(Sensor):
    def __init__(self, game, agent):
        super().__init__(game=game, agent=agent)
        self._nearest_mob = None

    @property
    def nearest_mob(self) -> pg.sprite.Sprite:
        return self._nearest_mob

    @property
    def nearest_mob_dist(self) -> pg.sprite.Sprite:
        return (
            calculate_point_dist(point1=self.agent.pos, point2=self.nearest_mob.pos)
            if self.nearest_mob
            else None
        )

    def _find_nearest_mob(self) -> None:
        def is_closer(mob, dist: int) -> bool:
            return (mob is not self.nearest_mob) and (dist < self.nearest_mob_dist)

        def set_nearest_mob(new_mob) -> None:
            if self.nearest_mob is not None:
                self._nearest_mob.is_nearest_mob = False
            self._nearest_mob = new_mob
            self._nearest_mob.is_nearest_mob = True

        for mob in self.game.mobs.sprites():
            dist = calculate_point_dist(point1=self.agent.pos, point2=mob.pos)
            if self.nearest_mob is None or is_closer(mob=mob, dist=dist):
                set_nearest_mob(new_mob=mob)

    def draw(self) -> None:
        self._find_nearest_mob()
        if self.nearest_mob:
            pg.draw.line(
                surface=self.game.screen,
                color=BLACK,
                start_pos=self.agent.pos,
                end_pos=self.nearest_mob.pos,
                width=2,
            )

    def update(self):
        return super().update()
