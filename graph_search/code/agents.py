from abc import abstractmethod

from pygame import (
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_a,
    K_d,
    K_s,
    K_w,
    Surface,
    Vector2,
    key,
    sprite,
    transform,
)

from config import AGENT_HIT_RECT, AGENT_ROT_SPEED, AGENT_SPEED, GREEN, TILESIZE


class Agent(sprite.Sprite):
    def __init__(self, game, x, y):
        self.game = game

        # PyGame-specific attributes
        self.groups = game.all_sprites
        sprite.Sprite.__init__(self, self.groups)

        # Position and movement attributes
        self.pos = Vector2(x, y) * TILESIZE
        self.vel = Vector2(0, 0)
        self.rot = 0

        self.draw()

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
        self.hit_rect = AGENT_HIT_RECT
        self.hit_rect.center = self.rect.center

    def collision(self, x: bool = None, y: bool = None):
        pass

    def draw(self):
        # Agent's visual attributes
        self.image = Surface((TILESIZE, TILESIZE))
        self.image.fill(color=GREEN)
        self.rect = self.image.get_rect()

    def move(self):
        self.rot_speed = 0
        self.vel = Vector2(0, 0)
        keys = key.get_pressed()

        if keys[K_LEFT] or keys[K_a]:
            self.rot_speed = AGENT_ROT_SPEED
        if keys[K_RIGHT] or keys[K_d]:
            self.rot_speed = -AGENT_ROT_SPEED
        if keys[K_UP] or keys[K_w]:
            self.vel = Vector2(AGENT_SPEED, 0).rotate(-self.rot)
        if keys[K_DOWN] or keys[K_s]:
            self.vel = Vector2(-AGENT_SPEED / 2, 0).rotate(-self.rot)

    def update(self):
        # Handle move keys being pressed
        self.move()
        self.rot = (self.rot + self.rot_speed * self.game.dt) % 360
        # Adjust image based on rotation
        self.image = transform.rotate(surface=self.game.agent_image, angle=self.rot)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        # Update position
        self.pos += self.vel * self.game.dt
        self.hit_rect.centerx = self.pos.x
        # Handle collisions
        self.collision("x")
        self.hit_rect.centery = self.pos.y
        self.collision("y")
        self.rect.center = self.hit_rect.center
