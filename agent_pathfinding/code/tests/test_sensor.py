from config import TILESIZE
from game import Game
from goals import Goal
from pygame import Vector2


def get_goal_objects(tmxdata) -> dict:
    """Return a dictionary of Goal obj's direction (NSEW) and position (x,y tuple)"""

    goals = {"north": (), "south": (), "east": (), "west": ()}

    goal_objs = [obj for obj in tmxdata.objects if obj.type == "goal"]
    for tile_object in goal_objs:
        x = tile_object.x
        y = tile_object.y
        goals[tile_object.properties["direction"]] = Vector2(x, y)

    return goals


class TestSensor:
    game = Game(manual=True, map_name="map_test_sensor.tmx")
    game.new()
    goals = get_goal_objects(tmxdata=game.map.tmxdata)

    def test_sensor_angle_to_nsew(self):
        """Place Goals NSEW of the Agent and verify angle_to are multiples of 90 deg"""
        assert self.game._map_name == "map_test_sensor.tmx"
        expected_angles = {"north": 90, "south": 270, "east": 0, "west": 180}

        agent = self.game.agent

        for direction, position in self.goals.items():
            Goal(game=self.game, x=position.x, y=position.y)
            self.game._update()
            angle = agent.goal_sensor.angle_to
            assert angle == expected_angles[direction], (
                f"{direction} angle {angle} does not match expected angle "
                f"{expected_angles[direction]} "
            )
            self.game.goals.empty()

    def test_sensor_angle_to_corners(self):
        """Place Goals in corner of the Agent and verify angle_to are multiples of 90+-45 deg"""
        assert self.game._map_name == "map_test_sensor.tmx"
        goals = {
            "northeast": Vector2(7, 1) * TILESIZE,
            "northwest": Vector2(1, 1) * TILESIZE,
            "southeast": Vector2(7, 7) * TILESIZE,
            "southwest": Vector2(1, 7) * TILESIZE,
        }
        expected_angles = {
            "northeast": 90 - 45,
            "northwest": 90 + 45,
            "southeast": 270 + 45,
            "southwest": 270 - 45,
        }

        agent = self.game.agent

        for direction, position in goals.items():
            Goal(game=self.game, x=position.x, y=position.y)
            self.game._update()
            angle = agent.goal_sensor.angle_to
            assert angle == expected_angles[direction], (
                f"{direction} angle {angle} does not match expected angle "
                f"{expected_angles[direction]} "
            )
            self.game.goals.empty()
