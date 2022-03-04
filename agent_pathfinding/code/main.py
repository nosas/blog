from game import Game

if __name__ == "__main__":
    g = Game(manual=True)

    while True:
        g.new()
        g.run()
        g.quit()
