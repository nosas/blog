from os import path
from sys import exit

import pygame as pg

from agents import AgentManual, Mob
from config import (
    AGENT_IMG,
    AGENT_RANDOM_SPAWN,
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
from goals import Goal
from map import TiledMap
from objects import Wall, Path, Sidewalk
from random import choice, random, randrange


class Game:
    def __init__(self):
        """Initialize the screen, game clock, and load game data"""
        pg.init()
        self.debug = DEBUG
        self.screen = pg.display.set_mode(size=(WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self._load_data()

    def _draw(self):
        """Draw all game images to screen: sprites, roads, paths, debug info"""
        # Draw the map
        self.screen.blit(source=self.map_img, dest=self.map_rect)

        # Draw all sprites' `image` attributes
        self.all_sprites.draw(surface=self.screen)
        self.paths.draw(surface=self.screen)
        if self.debug:
            self._draw_debug()
        pg.display.update()

    def _draw_debug(self):
        # Agent-specific debug outputs
        pg.draw.rect(self.screen, WHITE, self.agent.hit_rect, 0)
        self._draw_grid()

        # Draw green rectangles around sidewalks
        for sidewalk in self.sidewalks:
            pg.draw.rect(self.screen, GREEN, sidewalk.rect, 3)

        # Draw white rectangles around walls
        for wall in self.walls:
            pg.draw.rect(self.screen, WHITE, wall.hit_rect, 3)

        # Draw yellow rectangles over Path objects
        for p in self.paths:
            temp_rect = pg.Rect(p.x, p.y, p.rect.width, p.rect.height)
            pg.draw.rect(self.screen, YELLOW, temp_rect, 2)

    def _draw_fps():
        """Draw the FPS count"""
        pass

    def _draw_grid(self):
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

    def _events(self):
        """Handle key buttons, mouse clicks, etc."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._quit()
            if event.type == pg.KEYDOWN:
                if event.key in [pg.K_ESCAPE, pg.K_q]:
                    self._quit()
                if event.key == pg.K_r:
                    self.new()
                if event.key == pg.K_SPACE:
                    self.debug = not self.debug
                    print("agent pos   ", self.agent.pos)
                    print("agent posxy ", self.agent.pos.x, self.agent.pos.y)
                    print("agent rect  ", self.agent.rect.x, self.agent.rect.y)
                    print("agent hrect ", self.agent.hit_rect.x, self.agent.hit_rect.y)

    def _load_data(self):
        """Load game, image, and map data"""
        game_folder = path.dirname(__file__)
        img_folder = path.join(game_folder, "img")
        map_folder = path.join(img_folder, "map")

        self.map = TiledMap(filename=path.join(map_folder, MAP))
        self.map_img = self.map.make_map()
        self.map_rect = self.map_img.get_rect()

        self.agent_img = pg.image.load(path.join(img_folder, AGENT_IMG)).convert_alpha()
        self.agent_img = pg.transform.flip(
            surface=self.agent_img, flip_x=True, flip_y=False
        )
        self.agent_img = pg.transform.scale(
            surface=self.agent_img, size=(TILESIZE, TILESIZE)
        )

    def _quit(self):
        pg.quit()
        exit()

    def _update(self):
        """Update all sprite interactions"""
        self.all_sprites.update()

    def new(self):
        """Create sprite groups and convert tiles into game objects"""
        # PyGame object containers
        self.all_sprites = pg.sprite.Group()
        self.mobs = pg.sprite.Group()
        self.roads = pg.sprite.Group()
        self.paths = pg.sprite.Group()
        self.sidewalks = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.goals = pg.sprite.Group()

        # ! Object conversions will be moved to the classes' own methods
        for tile_object in self.map.tmxdata.objects:
            name = tile_object.name
            type = tile_object.type
            x = tile_object.x
            y = tile_object.y
            height = tile_object.height
            width = tile_object.width

            # if tile_object.type == "agent" and tile_object.name == "agent":
            if name == "agent":
                offset = TILESIZE / 2  # Center Agent's position to tile's center
                rot = 0
                if AGENT_RANDOM_SPAWN:
                    # TODO Verify Agent doesn't spawn on mob, battle, agent, tp, door
                    road = choice(self.roads.sprites())
                    x = road.x
                    y = road.y
                    rot = random() * 360
                self.agent = AgentManual(game=self, x=x + offset, y=y + offset, rot=rot)
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

        while len(self.mobs.sprites()) < NUM_MOBS:
            Mob(game=self, path=choice(self.paths.sprites()))

        random_road = choice(self.roads.sprites())
        Goal(
            game=self,
            x=random_road.x + randrange(random_road.rect.width),
            y=random_road.y + randrange(random_road.rect.height),
        )

    def run(self):
        """Start the game"""
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS)
            self._events()
            self._update()
            self._draw()
