import pygame as pg
import numpy as np


def calculate_point_dist(point1, point2) -> float:
    """Calculate the distance between two points"""
    # if type(point1) == pg.math.Vector2:
    point1 = np.asarray(point1)
    # if type(point2) == pg.math.Vector2:
    point2 = np.asarray(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))


def collide_hit_rect(obj_with_rect1, obj_with_rect2):
    return obj_with_rect1.hit_rect.colliderect(obj_with_rect2.rect)


def collide_with_walls(
    sprite: pg.sprite.Sprite, group: pg.sprite.Group, direction: str
):
    """Check if a sprite (Agent) collides with a Wall

    Args:
        sprite (pg.sprite.Sprite): Typically an Agent object
        group (pg.sprite.Group): Typically a Wall group
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