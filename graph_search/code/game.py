from sys import exit
from os import path
import pygame as pg
from config import AGENT_IMG, BG_COLOR, FPS, HEIGHT, LIGHTGREY, TILESIZE, TITLE, WIDTH
from agents import AgentManual


class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode(size=(WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()

        # PyGame object containers
        self.all_sprites = pg.sprite.Group()
        self.agent = AgentManual(game=self, x=10, y=10)
        self.load_data()

    def load_data(self):
        game_folder = path.dirname(__file__)
        img_folder = path.join(game_folder, "img")
        self.agent_image = pg.image.load(
            path.join(img_folder, AGENT_IMG)
        ).convert_alpha()
        self.agent_image = pg.transform.flip(
            surface=self.agent_image, flip_x=True, flip_y=False
        )

    def run(self):
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        exit()

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN and event.key in [pg.K_ESCAPE, pg.K_q]:
                self.quit()

    def update(self):
        self.all_sprites.update()

    def draw(self):
        self.screen.fill(color=BG_COLOR)
        self.draw_grid()
        self.all_sprites.draw(surface=self.screen)
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
