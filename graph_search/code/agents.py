from pygame import sprite
from abc import abstractmethod


class Agent(sprite.Sprite):
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
    pass
