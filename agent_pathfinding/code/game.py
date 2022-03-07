from os import path
from random import choice, random
from sys import exit

import numpy as np
import pygame as pg

from agents import AgentAuto, AgentManual, Mob
from config import (
    AGENT_IMG,
    AGENT_RANDOM_SPAWN,
    BLACK,
    DEBUG,
    FPS,
    GREEN,
    HEIGHT,
    LIGHTGREY,
    MAP,
    NUM_MOBS,
    TILESIZE,
    TITLE,
    WHITE,
    WIDTH,
    YELLOW,
)
from goals import Teleport
from helper import calculate_point_dist
from map import TiledMap
from objects import Path, Sidewalk, Wall
from typing import List, Tuple


class Game:
    # ! Prevent black from formatting this dictionary like a maniac
    # fmt: off
    event_ids = {
        1: ("w", pg.K_w),
        2: ("a", pg.K_a,),
        3: ("s", pg.K_s,),
        4: ("d", pg.K_d,),
        5: [("w", pg.K_w,), ("a", pg.K_a,)],
        6: [("w", pg.K_w,), ("d", pg.K_d,)],
        7: [("s", pg.K_s,), ("a", pg.K_a,)],
        8: [("s", pg.K_s,), ("d", pg.K_d,)],
        9: [("a", pg.K_a,), ("d", pg.K_d,)],
    }
    # fmt: on
    event_dict = {
        pg.K_w: {"unicode": "w", "key": 119, "mod": 0, "scancode": 26, "window": None},
        pg.K_a: {"unicode": "a", "key": 97, "mod": 0, "scancode": 4, "window": None},
        pg.K_s: {"unicode": "s", "key": 115, "mod": 0, "scancode": 22, "window": None},
        pg.K_d: {"unicode": "d", "key": 100, "mod": 0, "scancode": 7, "window": None},
    }

    def __init__(self, manual: bool = False, map_name: str = MAP):
        """Initialize the screen, game clock, and load game data"""

        self._agent_type = AgentManual if manual else AgentAuto
        self._map_name = map_name

        # Initialize pygame display and clock
        pg.init()
        self.screen = pg.display.set_mode(size=(WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self._load_data()

        # Booleans for drawing
        self.debug = DEBUG
        self.draw_sensors = True

    def _draw(self) -> None:
        """Draw all game images to screen: sprites, roads, paths, debug info"""
        # Draw the map
        self.screen.blit(source=self.map_img, dest=self._map_rect)

        # Draw all sprites' `image` and `rect` attributes
        self.all_sprites.draw(surface=self.screen)
        self.paths.draw(surface=self.screen)
        # self.agent.sensor.draw()
        # ! Temp for loop until I make proper rectangles for sensor lines
        if self.draw_sensors:
            [sensor.draw() for sensor in self.sensors]
        if self.debug:
            self._draw_debug()
        self._draw_game_info()
        pg.display.update()

    def _draw_debug(self) -> None:
        # Agent-specific debug outputs
        pg.draw.rect(self.screen, WHITE, self.agent.hit_rect, 0)
        self._draw_grid()

        # Draw green rectangles around sidewalks
        for sidewalk in self.sidewalks:
            pg.draw.rect(self.screen, GREEN, sidewalk.rect, 3)

        # Draw white rectangles around walls
        for wall in self.walls:
            pg.draw.rect(self.screen, WHITE, wall.rect, 3)

        # Draw yellow rectangles over Path objects
        for p in self.paths:
            temp_rect = pg.Rect(p.x, p.y, p.rect.width, p.rect.height)
            pg.draw.rect(self.screen, YELLOW, temp_rect, 2)

    def _draw_grid(self) -> None:
        """Draw a screen-wide grid"""
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(
                surface=self.screen,
                color=LIGHTGREY,
                start_pos=(x, 0),
                end_pos=(x, HEIGHT),
            )
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(
                surface=self.screen,
                color=LIGHTGREY,
                start_pos=(0, y),
                end_pos=(WIDTH, y),
            )

    def _draw_game_info(self) -> None:
        """Draw Agent's coords, velocity, sensors"""
        # ! We need a cleaner way to display text, this is getting out of control

        box = pg.Rect(25, 25, 300, 150)
        pg.draw.rect(surface=self.screen, color=WHITE, rect=box)

        font = pg.font.SysFont("monospace", 16)

        agent_pos = np.asarray(self.agent.pos)
        text_agent_pos = font.render(
            f"Agent Position: {(agent_pos/TILESIZE).astype('int')}",
            False,
            BLACK,
        )
        self.screen.blit(text_agent_pos, (box.x + 5, box.y))

        mouse_pos = np.asarray(pg.mouse.get_pos())
        text_mouse_pos = font.render(
            f"Mouse Position: {(mouse_pos/TILESIZE).astype('int')}", False, BLACK
        )
        self.screen.blit(
            text_mouse_pos, (box.x + 5, box.y + text_agent_pos.get_height() * 1)
        )

        # mouse_agent_dist_tuple = tuple(map(sub, agent_pos, mouse_pos))
        mouse_agent_dist = calculate_point_dist(mouse_pos, agent_pos)
        text_mouse_agent_dist = font.render(
            f"Mouse Distance: {mouse_agent_dist/TILESIZE:.1f}", False, BLACK
        )
        self.screen.blit(
            text_mouse_agent_dist, (box.x + 5, box.y + text_agent_pos.get_height() * 2)
        )

        if self.agent.nearest_mob:  # ! Required to prevent race condition in game.new()
            text_nearest_mob_dist = font.render(
                f"Near Mob Dist : {self.agent.mob_sensor.dist/TILESIZE:.1f}",
                False,
                BLACK,
            )
            self.screen.blit(
                text_nearest_mob_dist,
                (box.x + 5, box.y + text_agent_pos.get_height() * 3),
            )

        if self.agent.nearest_goal:
            goal_dist = f"{self.agent.goal_sensor.dist/TILESIZE:.1f}"
            goal_pos = np.asarray(self.agent.nearest_goal.pos)
            goal_pos = f"{(goal_pos/TILESIZE).astype('int')}"
            text_nearest_goal_dist = font.render(
                f"Near Goal Dist: {goal_dist} {goal_pos}",
                False,
                BLACK,
            )
            self.screen.blit(
                text_nearest_goal_dist,
                (box.x + 5, box.y + text_agent_pos.get_height() * 4),
            )

        text_agent_dist_traveled = font.render(
            f"Distance Moved: {self.agent.distance_traveled/TILESIZE:.1f}", False, BLACK
        )
        self.screen.blit(
            text_agent_dist_traveled,
            (box.x + 5, box.y + text_agent_pos.get_height() * 5),
        )

    def _events(self) -> None:
        """Handle key buttons, mouse clicks, etc."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            elif event.type == pg.KEYDOWN:
                if type(self.agent) == AgentAuto:
                    if event.key in [pg.K_w, pg.K_a, pg.K_s, pg.K_d]:
                        self.agent.move(key=event.key)
                if event.key in [pg.K_ESCAPE, pg.K_q]:
                    self.quit()
                elif event.key == pg.K_r:
                    self.new()
                elif event.key == pg.K_SPACE:
                    self.debug = not self.debug
                    print("agent pos   ", self.agent.pos)
                    print("agent posxy ", self.agent.pos.x, self.agent.pos.y)
                    print("agent rect  ", self.agent.rect.x, self.agent.rect.y)
                    print("agent hrect ", self.agent.hit_rect.x, self.agent.hit_rect.y)
                elif event.key == pg.K_1:
                    self.draw_sensors = not self.draw_sensors
            elif event.type == pg.MOUSEBUTTONUP and event.button == 3:  # RIGHT button
                self.agent.pos = pg.mouse.get_pos()

    def _load_data(self) -> None:
        """Load game, image, and map data"""
        game_folder = path.dirname(__file__)
        img_folder = path.join(game_folder, "img")

        self._load_map(map_name=self._map_name)

        self.agent_img = pg.image.load(path.join(img_folder, AGENT_IMG)).convert_alpha()
        self.agent_img = pg.transform.flip(
            surface=self.agent_img, flip_x=True, flip_y=False
        )
        self.agent_img = pg.transform.scale(
            surface=self.agent_img, size=(TILESIZE, TILESIZE)
        )

    def _load_map(self, map_name) -> None:
        game_folder = path.dirname(__file__)
        img_folder = path.join(game_folder, "img")
        map_folder = path.join(img_folder, "map")

        self.map = TiledMap(filename=path.join(map_folder, map_name))
        self.map_img = self.map.make_map()
        self._map_name = map_name
        self._map_rect = self.map_img.get_rect()

    def _post_event(self, events: List[Tuple[str, pg.event.Event]]) -> None:
        """Add a custom event (key press) to the Game's Event queue"""
        if type(events) == tuple:
            events = [events]

        for unicode, event_id in events:
            # print(f"Pressed {unicode}")
            event = pg.event.Event(pg.KEYDOWN, self.event_dict[event_id])
            pg.event.post(event)
            event = pg.event.Event(pg.KEYUP, self.event_dict[event_id])
            pg.event.post(event)

    def _update(self, action: int = None) -> None:
        """Update all sprite interactions"""
        self.dt = self.clock.tick(FPS)
        if action:
            events = self.event_ids[action]
            self._post_event(events=events)
        self._events()
        self.all_sprites.update()
        self._draw()

    def new(self) -> None:
        """Create sprite groups and convert tiles into game objects"""
        self.playing = True

        # PyGame object containers
        self.all_sprites = pg.sprite.Group()
        self.mobs = pg.sprite.Group()
        self.roads = pg.sprite.Group()
        self.paths = pg.sprite.Group()
        self.sidewalks = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.goals = pg.sprite.Group()
        self.sensors = pg.sprite.Group()

        # ! Object conversions will be moved to the classes' own methods
        for tile_object in self.map.tmxdata.objects:
            if not tile_object.visible:
                continue
            name = tile_object.name
            type = tile_object.type
            x = tile_object.x
            y = tile_object.y
            height = tile_object.height
            width = tile_object.width

            # if tile_object.type == "agent" and tile_object.name == "agent":
            if name == "agent":
                offset = TILESIZE / 2  # Center Agent's position to tile's center
                heading = 0
                if AGENT_RANDOM_SPAWN:
                    # TODO Verify Agent doesn't spawn on mob, battle, agent, tp, door
                    road = choice(self.roads.sprites())
                    x = road.x
                    y = road.y
                    heading = random() * 360
                self.agent = self._agent_type(
                    game=self, x=x + offset, y=y + offset, heading=heading
                )
            elif type == "road":
                if name == "sidewalk":
                    Sidewalk(game=self, x=x, y=y, width=width, height=height)
                if name == "path":
                    Path(
                        game=self,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        direction=tile_object.properties["direction"],
                    )
            elif type == "wall":
                Wall(
                    game=self,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
            elif type == "goal":
                if name == "teleport":
                    Teleport(game=self, x=x, y=y)

        while len(self.mobs.sprites()) < NUM_MOBS:
            Mob(game=self, path=choice(self.paths.sprites()))

    def quit(self) -> None:
        pg.quit()
        exit()

    def run(self) -> None:
        """Start the game"""
        while self.playing:
            self._update()
