from abc import abstractmethod
from pathlib import Path

import pygame as pg

from config import (
    AGENT_HIT_RECT,
    AGENT_ROT_SPEED,
    AGENT_SPEED,
    RED,
    MOB_SPEED,
    MOB_SIZE,
)


def collide_hit_rect(obj_with_rect1, obj_with_rect2):
    return obj_with_rect1.hit_rect.colliderect(obj_with_rect2.rect)


def collide_with_walls(
    sprite: pg.sprite.Sprite, group: pg.sprite.Group, direction: str
):
    """Check if a sprite (Agent) collides with some other Group (Wall)

    Args:
        sprite (pg.sprite.Sprite): Typically an Agent object
        group (pg.sprite.Group): Typically a Wall object
        direction (str): "x" or "y"
    """
    if direction == "x":
        hits = pg.sprite.spritecollide(
            sprite=sprite, group=group, dokill=False, collided=collide_hit_rect
        )
        if hits:
            if hits[0].rect.centerx > sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.left - sprite.hit_rect.width / 2
            if hits[0].rect.centerx < sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.right + sprite.hit_rect.width / 2
            sprite.vel.x = 0
            sprite.hit_rect.centerx = sprite.pos.x
    if direction == "y":
        hits = pg.sprite.spritecollide(
            sprite=sprite, group=group, dokill=False, collided=collide_hit_rect
        )
        if hits:
            if hits[0].rect.centery > sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.top - sprite.hit_rect.height / 2
            if hits[0].rect.centery < sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.bottom + sprite.hit_rect.height / 2
            sprite.vel.y = 0
            sprite.hit_rect.centery = sprite.pos.y


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x: float, y: float):
        self.game = game

        # PyGame-specific attributes
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)

        # Position and movement attributes
        self.pos = pg.Vector2(x, y)  # * TILESIZE
        self.vel = pg.Vector2(0, 0)
        self.rot = 0

    @abstractmethod
    def _move(self):
        """Determine which keys are being pressed and handle Agent movement"""
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    @abstractmethod
    def _collision(self, direction):
        """Handle Agent reactions when colliding with walls or other Agents"""
        raise NotImplementedError(
            f"Class {self.__class__} does not have a collision method"
        )

    @abstractmethod
    def draw(self):
        """How the agent is drawn on the map"""
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    @abstractmethod
    def update(self):
        """All required checks/updates made to the Agent on every Game tick"""
        raise NotImplementedError(
            f"Class {self.__class__} does not have an update method"
        )


class AgentManual(Agent):
    def __init__(self, game, x: float, y: float, rot: int = 0):
        super().__init__(game=game, x=x, y=y)
        self.image = game.agent_img
        self.rot = rot
        self.rect = self.image.get_rect()
        self.hit_rect = AGENT_HIT_RECT
        self.hit_rect.center = self.rect.center

    def _collision(self, x: bool = None, y: bool = None):
        assert not (x is not None and y is not None)
        return collide_with_walls(
            sprite=self, group=self.game.walls, direction="x" if x else "y"
        )

    def _move(self):
        self.rot_speed = 0
        self.vel = pg.Vector2(0, 0)
        keys = pg.key.get_pressed()

        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.rot_speed = AGENT_ROT_SPEED
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.rot_speed = -AGENT_ROT_SPEED
        if keys[pg.K_UP] or keys[pg.K_w]:
            self.vel = pg.Vector2(AGENT_SPEED, 0).rotate(-self.rot)
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.vel = pg.Vector2(-AGENT_SPEED / 2, 0).rotate(-self.rot)

    def draw(self):
        # Agent's visual attributes are loaded in `game.py` and updated in self.update()
        # self.image = Surface((TILESIZE, TILESIZE))
        # self.rect = self.image.get_rect()
        pass

    def update(self):
        """Moves, rotates, and updates the agent's position and image. Also checks for collisions"""
        # Handle move keys being pressed
        self._move()
        self.rot = (self.rot + self.rot_speed * self.game.dt) % 360

        # Adjust image based on rotation
        self.image = pg.transform.rotate(surface=self.game.agent_img, angle=self.rot)
        self.rect = self.image.get_rect()

        # Update position
        self.pos += self.vel * self.game.dt

        # Handle collisions
        self.hit_rect.centerx = self.pos.x
        self._collision(x=True)
        self.hit_rect.centery = self.pos.y
        self._collision(y=True)
        self.rect.center = self.hit_rect.center


class Mob(pg.sprite.Sprite):
    def __init__(self, game, x: float, y: int, path: Path):
        self.game = game
        self.groups = game.all_sprites, game.mobs
        pg.sprite.Sprite.__init__(self, self.groups)

        self.pos = pg.Vector2(x, y)  # * TILESIZE
        self.vel = pg.Vector2(0, 0)
        self.path = path
        self.direction = path.direction

        self.image = pg.Surface(MOB_SIZE)
        self.image.fill(color=RED)
        self.rect = self.image.get_rect()
        self._align_with_path()
        self.rect.center = self.pos
        self.hit_rect = AGENT_HIT_RECT
        self.hit_rect.center = self.rect.center

    def _align_with_path(self):
        # Adjust self.rect.center to be in the middle of the path
        if self.path.direction in ["left", "right"]:
            self.pos.y = self.path.rect.centery
        elif self.path.direction in ["up", "down"]:
            self.pos.x = self.path.rect.centerx
        self.rect.center = self.pos

    def _move(self):
        vel = {
            "left": pg.Vector2(-MOB_SPEED, 0),
            "right": pg.Vector2(MOB_SPEED, 0),
            "up": pg.Vector2(0, -MOB_SPEED),
            "down": pg.Vector2(0, MOB_SPEED),
        }
        self.vel = vel[self.direction]

    def update(self):
        """Moves, rotates, and updates the agent's position and image. Also checks for collisions"""
        # Handle move keys being pressed
        self._move()
        # Update position
        self._align_with_path()
        self.pos += self.vel * self.game.dt
        self.rect.center = self.pos

        paths = pg.sprite.spritecollide(
            sprite=self, group=self.game.paths, dokill=False
        )
        if len(paths) == 1 and paths[0] != self.path:
            self.path = paths[0]
            self.direction = self.path.direction
