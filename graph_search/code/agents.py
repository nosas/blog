from abc import abstractmethod

import pygame as pg

from config import AGENT_HIT_RECT, AGENT_ROT_SPEED, AGENT_SPEED


def collide_hit_rect(obj1, obj2):
    return obj1.hit_rect.colliderect(obj2.rect)


def collide_with_walls(sprite, group, dir):
    if dir == "x":
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            if hits[0].rect.centerx > sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.left - sprite.hit_rect.width / 2
            if hits[0].rect.centerx < sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.right + sprite.hit_rect.width / 2
            sprite.vel.x = 0
            sprite.hit_rect.centerx = sprite.pos.x
    if dir == "y":
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            if hits[0].rect.centery > sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.top - sprite.hit_rect.height / 2
            if hits[0].rect.centery < sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.bottom + sprite.hit_rect.height / 2
            sprite.vel.y = 0
            sprite.hit_rect.centery = sprite.pos.y


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.game = game

        # PyGame-specific attributes
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)

        # Position and movement attributes
        self.pos = pg.Vector2(x, y)  # * TILESIZE
        self.vel = pg.Vector2(0, 0)
        self.rot = 0

    @abstractmethod
    def draw(self):
        """How the agent is drawn on the map"""
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    @abstractmethod
    def move(self):
        """Determine which keys are being pressed and handle Agent movement"""
        raise NotImplementedError(f"Class {self.__class__} does not have a move method")

    @abstractmethod
    def collision(self, direction):
        """Handle Agent reactions when colliding with walls or other Agents"""
        raise NotImplementedError(
            f"Class {self.__class__} does not have a collision method"
        )

    @abstractmethod
    def update(self):
        """All required checks/updates made to the Agent on every Game tick"""
        raise NotImplementedError(
            f"Class {self.__class__} does not have an update method"
        )


class AgentManual(Agent):
    def __init__(self, game, x: int, y: int):
        super().__init__(game=game, x=x, y=y)
        self.image = game.agent_img
        self.rect = self.image.get_rect()
        self.hit_rect = AGENT_HIT_RECT
        self.hit_rect.center = self.rect.center

    def collision(self, x: bool = None, y: bool = None):
        pass

    def draw(self):
        # Agent's visual attributes
        # self.image = Surface((TILESIZE, TILESIZE))
        # self.rect = self.image.get_rect()
        pass

    def move(self):
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

    def update(self):
        # Handle move keys being pressed
        self.move()
        self.rot = (self.rot + self.rot_speed * self.game.dt) % 360

        # Adjust image based on rotation
        self.image = pg.transform.rotate(surface=self.game.agent_img, angle=self.rot)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

        # Update position
        self.pos += self.vel * self.game.dt
        self.hit_rect.centerx = self.pos.x

        # Handle collisions
        self.collision(x=True)
        self.hit_rect.centery = self.pos.y
        self.collision(y=True)
        self.rect.center = self.hit_rect.center
