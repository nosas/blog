from sys import exit

import pygame as pg

from config import BG_COLOR, FPS, HEIGHT, LIGHTGREY, TILESIZE, TITLE, WIDTH


class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode(size=(WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()

    def run(self):
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    def stop(self):
        self.playing = False

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
        pass

    def draw(self):
        self.screen.fill(color=BG_COLOR)
        self.draw_grid()
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
