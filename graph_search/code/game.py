from sys import exit
from os import path
import pygame as pg
from config import (
    AGENT_IMG,
    WHITE,
    FPS,
    HEIGHT,
    LIGHTGREY,
    MAP,
    TILESIZE,
    TITLE,
    WIDTH,
)
from agents import AgentManual
from map import TiledMap, Path


class Game:
    def __init__(self):X
        pg.init()
        self.screen = pg.display.set_mode(size=(WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self._load_data()

    def _load_data(self):
        game_folder = path.dirname(__file__)
        img_folder = path.join(game_folder, "img")cc
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

    def new(self):
        # PyGame object containers
        self.all_sprites = pg.sprite.Group()
        self.roads = pg.sprite.Group()
        self.walls = pg.sprite.Group()

        for tile_object in self.map.tmxdata.objects:
            # if tile_object.type == "agent" and tile_object.name == "agent":
            if tile_object.name == "agent":
                print(tile_object.x, tile_object.y)
                self.agent = AgentManual(game=self, x=tile_object.x, y=tile_object.y)
                print(self.agent.pos.x, self.agent.pos.y)
            if tile_object.name == "path":
                Path(
                    game=self,
                    x=tile_object.x,
                    y=tile_object.y,
                    width=tile_object.width,
                    height=tile_object.height,
                    direction=tile_object.properties["direction"],
                )
            if tile_object.name == "teleport":
                self.img_bitmap = self.map.tmxdata.get_tile_image_by_gid(
                    tile_object.gid
                )
                self.temp_rect = pg.Rect(
                    tile_object.x, tile_object.y, tile_object.width, tile_object.height
                )

    def run(self):
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.quit()
                if event.type == pg.KEYDOWN and event.key in [pg.K_ESCAPE, pg.K_q]:
                    self.quit()

            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        exit()

    def update(self):
        self.all_sprites.update()
        self.roads.update()

    def draw(self):
        self.screen.blit(source=self.map_img, dest=self.map_rect)
        # self.draw_grid()
        self.all_sprites.draw(surface=self.screen)
        self.roads.draw(surface=self.screen)
        for p in self.roads:
            self.screen.blit(source=p.image, dest=(p.x, p.y))
        pg.display.update()

    def draw_grid(self):
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

    def draw_fps():
        pass
