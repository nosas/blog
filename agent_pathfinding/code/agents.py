from __future__ import annotations

from abc import abstractmethod
from random import choice, randrange

import pygame as pg

from config import (
    AGENT_HIT_RECT,
    AGENT_ROT_SPEED,
    AGENT_SPEED,
    BLACK,
    BLUE,
    BROWN,
    GREEN,
    MOB_SIZE,
    MOB_SPEED,
    ORANGE,
    RED,
    TILESIZE,
    YELLOW,
)
from goals import Goal
from helper import (
    angle_is_between,
    calculate_point_dist,
    collide_hit_rect,
    collide_with_walls,
)
from objects import Path
from sensor import CardinalSensor, ObjectSensor
from typing import List, Tuple


def _collision_with_mobs(sprite: pg.sprite.Sprite) -> List[pg.sprite.Sprite]:
    """Return None or a list of Mobs that are colliding with a sprite (Agent/Mob)

    If the Mob is not Battling:
        Agent: Both sprites engage in Battle
        Mob: Sprite de-spawns
    If the Mob is Battling:
        Agent: Joins Battle if Battle is not full
        Mob: Chance to join Battle or de-spawn

    Args:
        sprite (pg.sprite.Sprite): Agent or Mob Sprite object
    """

    def hit_rect_collision(sprite1, sprite2) -> bool:
        """Return bool of sprite1.hit_rect colliding with sprite2.hit_rect"""
        if sprite1 != sprite2:
            return sprite1.hit_rect.colliderect(sprite2.hit_rect)

    return pg.sprite.spritecollide(
        sprite=sprite, group=sprite.game.mobs, dokill=False, collided=hit_rect_collision
    )


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x: float, y: float, heading: int = 0):
        self.game = game

        # PyGame-specific attributes
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)

        # Position and movement attributes
        self.distance_traveled = 0
        self.pos = pg.Vector2(x, y)
        self.last_valid_pos = pg.Vector2(x, y)
        self.spawn = pg.Vector2(x, y)

        self.vel = pg.Vector2(0, 0)
        self.heading = heading
        self.hit_rect = AGENT_HIT_RECT.copy()

    @property
    def distance_traveled_from_spawn(self) -> float:
        """Return distance traveled (in tiles) from the Agent's spawn"""
        return calculate_point_dist(point1=self.pos, point2=self.spawn) / TILESIZE

    @property
    def tpos(self) -> pg.Vector2:
        """Return Agent's tile position instead of pixel (world) position"""
        return pg.Vector2(self.pos) / TILESIZE

    @abstractmethod
    def _move(self) -> None:
        """Handle key input and Agent movement"""
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    @abstractmethod
    def _collision(self, direction) -> None:
        """Handle Agent reactions when colliding with Walls, Goals, Mobs, or Agents"""
        raise NotImplementedError(
            f"Class {self.__class__} does not have a collision method"
        )

    @abstractmethod
    def draw(self) -> None:
        """How the agent is drawn on the map"""
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    @abstractmethod
    def update(self) -> None:
        """Update Agent's position, image, and collisions on every Game tick"""
        raise NotImplementedError(
            f"Class {self.__class__} does not have an update method"
        )


class AgentManual(Agent):
    """Agent to be manually controlled with WASD or arrow keys"""

    def __init__(self, game, x: float, y: float, heading: int = 0):
        super().__init__(game=game, x=x, y=y, heading=heading)
        self.image = game.agent_img
        self.rect = self.image.get_rect()
        self.hit_rect.center = self.rect.center
        self.battle = False

        self.goal_sensor = ObjectSensor(
            game=self.game, agent=self, group=self.game.goals, color=GREEN
        )
        self.mob_sensor = ObjectSensor(
            game=self.game, agent=self, group=self.game.mobs, color=BLACK
        )
        self.sensor = CardinalSensor(game=self.game, agent=self)

    @property
    def nearest_mob(self) -> Mob:
        return self.mob_sensor.nearest

    @property
    def nearest_goal(self) -> Goal:
        return self.goal_sensor.nearest if self.goal_sensor.nearest else self

    @property
    def is_headed_to_goal(self) -> bool:
        angle_range = 30  # FOV, Agent's field of view
        return angle_is_between(
            angle=self.heading,
            min_a=self.goal_sensor.angle_to - angle_range,
            max_a=self.goal_sensor.angle_to + angle_range,
        )

    @property
    def observation(self) -> dict:
        return {
            "dist_to_goal": self.goal_sensor.dist,
            "dist_traveled": self.distance_traveled,
            # "dist_traveled_from_spawn": self.distance_traveled_from_spawn,
            "heading": self.heading,
            "heading_goal": self.goal_sensor.angle_to,
            "is_battling": self.battle,
            "is_headed_to_goal": self.is_headed_to_goal,
            "is_hitting_wall": self.sensor.is_hitting_wall,
            "is_in_corner": self.sensor.is_in_corner,
            "tposx": self.tpos.x,
            "tposy": self.tpos.y,
            "delta_goal_tposx": self.nearest_goal.tpos.x - self.tpos.x,
            "delta_goal_tposy": self.nearest_goal.tpos.y - self.tpos.y,
            "cardinal_objs": self.sensor.objs,
            "cardinal_dists": self.sensor.dists,
        }

    def _collision(self) -> None:
        """Handle Agent reactions when colliding with Walls, Goals, Mobs, or Agents"""

        def collision_goal() -> None:
            """If Agent collides with Goal, end the Game"""
            goal = pg.sprite.spritecollide(
                sprite=self,
                group=self.game.goals,
                dokill=False,
                collided=collide_hit_rect,
            )
            if goal:
                print(f"Reached goal, traveled {self.distance_traveled:.2f}")
                # TODO Different interaction based on Goal type: Teleport, Mob, Door
                self.game.playing = False

        def collision_mob() -> None:
            """If Agent collides with any Mob, both enter Battle and cannot move"""
            mobs = _collision_with_mobs(sprite=self)
            if any(mobs) and not self.battle:
                print("Entering battle")
                self.battle = True
                mobs[0].battle = True
                self.game.playing = False

        def collision_wall() -> None:
            """Check for Wall collisions, prevent Agent from breaching Wall perimeter"""
            self.hit_rect.centerx = self.pos.x
            collide_with_walls(sprite=self, group=self.game.walls, direction="x")
            self.hit_rect.centery = self.pos.y
            collide_with_walls(sprite=self, group=self.game.walls, direction="y")

        collision_mob()
        collision_wall()
        collision_goal()

    def _move(self) -> None:
        """Handle key input and Agent movement"""
        self.heading_speed = 0
        self.vel = pg.Vector2(0, 0)
        keys = pg.key.get_pressed()

        # Not using if/elif so Agent can simultaneously press multiple keys
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.heading_speed = AGENT_ROT_SPEED
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.heading_speed = -AGENT_ROT_SPEED
        if keys[pg.K_UP] or keys[pg.K_w]:
            self.vel = pg.Vector2(AGENT_SPEED, 0).rotate(-self.heading)
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.vel = pg.Vector2(-AGENT_SPEED / 2, 0).rotate(-self.heading)

        self.heading = (self.heading + self.heading_speed * self.game.dt) % 360
        self.pos += self.vel * self.game.dt

    def draw(self) -> None:
        # Agent's visual attributes are loaded in `game.py` and updated in self.update()
        # Adjust image based on Agent's headingation
        self.image = pg.transform.rotate(
            surface=self.game.agent_img, angle=self.heading
        )
        self.rect = self.image.get_rect()

    def update(self) -> None:
        """Update Agent's position, image, and collisions on every Game tick"""
        if not self.battle:
            old_tpos = self.tpos
            # Update position
            self._move()
            # Adjust image
            self.draw()
            # Handle collisions
            self._collision()
            if self.game.map.is_a_tile(tx=self.tpos.x, ty=self.tpos.y):
                # Re-align image.rect to hit_rect
                self.rect.center = self.hit_rect.center
                self.last_valid_pos = pg.Vector2(self.pos.x, self.pos.y)
            else:
                # ! Fixes #16 Prevent Agent from escaping Map's perimeter in training
                self.pos = pg.Vector2(self.last_valid_pos.x, self.last_valid_pos.y)
                self.rect.center = self.pos
                self.hit_rect.center = self.rect.center
            # Accumulate distance traveled
            if not self.observation["is_in_corner"]:
                self.distance_traveled += calculate_point_dist(
                    point1=old_tpos, point2=self.tpos
                )


class AgentAuto(AgentManual):
    """Agent to be controlled by the OpenAI Gym environment"""

    def __init__(self, game, x: float, y: float, heading: int = 0):
        super().__init__(game, x, y, heading)
        self._moves = []

    def move(self, key: int) -> None:
        """Append a move (key press) to move list"""
        self._moves.append(key)

    def _move(self) -> None:
        """Handle key input and Agent movement"""
        self.heading_speed = 0
        self.vel = pg.Vector2(0, 0)

        # Not using if/elif so Agent can simultaneously press multiple keys
        for key in self._moves:
            if key in [pg.K_LEFT, pg.K_a]:
                self.heading_speed = AGENT_ROT_SPEED
            if key in [pg.K_RIGHT, pg.K_d]:
                self.heading_speed = -AGENT_ROT_SPEED
            if key in [pg.K_UP, pg.K_w]:
                self.vel = pg.Vector2(AGENT_SPEED, 0).rotate(-self.heading)
            if key in [pg.K_DOWN, pg.K_s]:
                self.vel = pg.Vector2(-AGENT_SPEED / 2, 0).rotate(-self.heading)

        self.heading = (self.heading + self.heading_speed * self.game.dt) % 360
        self.pos += self.vel * self.game.dt
        self._moves.clear()


class Mob(pg.sprite.Sprite):
    # Unique color for each Mob type
    _symbols = {"A": RED, "B": YELLOW, "C": ORANGE, "D": BROWN}

    @staticmethod
    def _get_symbol(mob_type: str) -> Tuple[int, int, int]:
        return Mob._symbols[mob_type]

    @property
    def _symbol(self) -> Tuple[int, int, int]:
        return (
            Mob._get_symbol(mob_type=self.mob_type) if not self.is_nearest_mob else BLUE
        )

    @property
    def heading(self) -> int:
        heading = {"up": 90, "down": 270, "left": 180, "right": 0}
        return heading[self.direction]

    def __init__(self, game, path: Path, mob_type: str = None):
        self.game = game
        self.groups = game.all_sprites, game.mobs
        pg.sprite.Sprite.__init__(self, self.groups)

        self.path = path
        x = randrange(self.path.x, self.path.rect.x + self.path.rect.width)
        y = randrange(self.path.y, self.path.rect.y + self.path.rect.height)

        self.pos = pg.Vector2(x, y)
        self.vel = pg.Vector2(0, 0)
        self.direction = path.direction
        self.is_nearest_mob = False

        self.mob_type = mob_type if mob_type else choice(list(Mob._symbols))
        self.image = pg.Surface(MOB_SIZE)
        self.image.fill(color=self._symbol)

        self.rect = self.image.get_rect()
        self.hit_rect = AGENT_HIT_RECT.copy()
        self._align_with_path()
        self.battle = False

    def _align_with_path(self) -> None:
        """Align Mob's position to the middle of the path"""
        if self.path.direction in ["left", "right"]:
            self.pos.y = self.path.rect.centery
        elif self.path.direction in ["up", "down"]:
            self.pos.x = self.path.rect.centerx
        self.hit_rect.center = self.rect.center = self.pos

    def _collision(self) -> None:
        """Handle collision with Mobs and Paths"""
        self._collision_mob()
        self._collision_path()

    def _collision_mob(self) -> None:
        """Despawn if collides with another Mob who's in Battle"""
        mobs = _collision_with_mobs(sprite=self)
        if any(mobs) and not self.battle:
            # TODO Add chance for self.battle = True
            self.despawn()

    def _collision_path(self) -> None:
        """Update Mob's current Path and align with the Path"""
        paths = pg.sprite.spritecollide(sprite=self, group=self.game.paths, dokill=0)
        if len(paths) == 1 and paths[0] != self.path:
            new_p = paths[0]
            # Update self.path only when the Mob reaches the new_path's centerx or centery
            # This is necessary to prevent Mobs teleporting from the Path's edge to Path's center
            if (
                (self.path.direction == "left" and self.pos.x <= new_p.rect.centerx)
                or (self.path.direction == "right" and self.pos.x >= new_p.rect.centerx)
                or (self.path.direction == "up" and self.pos.y <= new_p.rect.centery)
                or (self.path.direction == "down" and self.pos.y >= new_p.rect.centery)
            ):
                self.path = new_p
                self.direction = self.path.direction
                self._align_with_path()

    def _move(self) -> None:
        """Update Mob's velocity based on the Mob's direction"""
        vel = {
            "left": pg.Vector2(-MOB_SPEED, 0),
            "right": pg.Vector2(MOB_SPEED, 0),
            "up": pg.Vector2(0, -MOB_SPEED),
            "down": pg.Vector2(0, MOB_SPEED),
        }
        self.vel = vel[self.direction]

    def despawn(self) -> None:
        """Remove Mob from the Game"""
        self.kill()

    def update(self) -> None:
        """Update Mob's position, image, and collisions on every Game tick"""
        # Handle move keys being pressed
        if not self.battle:
            self._move()
            # Update position
            self.pos += self.vel * self.game.dt
            self.rect.center = self.pos
            self.hit_rect.center = self.rect.center
            # Update self.path when Mob collides with a new Path
            self._collision()
            # self.image.fill(color=self._symbol)
