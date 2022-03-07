from game import Game


class TestGame:
    g = Game(manual=True)

    def test_game_map_name_default(self):
        # TODO Assert goal tilepos
        assert self.g._map_name == "map16.tmx"

    def test_game_map_name_custom(self):
        # TODO Assert goal tilepos
        self.g._load_map(map_name="map_goal1_straight.tmx")
        assert self.g._map_name == "map_goal1_straight.tmx"
