"""Geometry tests run without Qt or a GUI application."""

import unittest

from src.core.map_coordinates import MapGeometry, check_frames, validate_map_metadata
from src.core.models import Waypoint


class TestMapCoordinates(unittest.TestCase):
    def test_resolution_005(self):
        geometry = MapGeometry(0.05, (-10, -20, 0), 500, 500)
        px, py = geometry.world_to_pixel(2.35, -1.12)
        self.assertAlmostEqual(px, 247)
        self.assertAlmostEqual(py, 122.4)
        x, y = geometry.pixel_to_world(247, 122.4)
        self.assertAlmostEqual(x, 2.35)
        self.assertAlmostEqual(y, -1.12)

    def test_resolution_010(self):
        geometry = MapGeometry(0.10, (-10, -20, 0), 500, 500)
        x, y = geometry.world_to_pixel(2.35, -1.12)
        self.assertAlmostEqual(x, 123.5)
        self.assertAlmostEqual(y, 311.2)

    def test_dimensions_origins_and_arbitrary_resolutions(self):
        for width, height in ((500, 500), (1000, 1000), (321, 879)):
            for resolution in (0.05, 0.10, 0.037, 2.5):
                with self.subTest(size=(width, height), resolution=resolution):
                    geometry = MapGeometry(resolution, (-2.317, 4.219, 0), width, height)
                    x, y = geometry.world_to_pixel(2.35123456789, -1.123456789)
                    self.assertAlmostEqual(x, (2.35123456789 + 2.317) / resolution)
                    self.assertAlmostEqual(y, height - (-1.123456789 - 4.219) / resolution)
                    wx, wy = geometry.pixel_to_world(x, y)
                    self.assertAlmostEqual(wx, 2.35123456789)
                    self.assertAlmostEqual(wy, -1.123456789)

    def test_image_boundaries_and_cell_centres(self):
        geometry = MapGeometry(0.1, (-1, -2, 0), 500, 1000)
        self.assertEqual(geometry.pixel_to_world(0, 1000), (-1, -2))
        self.assertEqual(geometry.world_to_pixel(-1, -2), (0, 1000))
        x, y = geometry.pixel_to_world(0.5, 999.5)
        self.assertAlmostEqual(x, -0.95)
        self.assertAlmostEqual(y, -1.95)

    def test_rebind_preserves_world_values_exactly(self):
        waypoint = Waypoint(0, 0, 0.785123456789)
        waypoint.set_world_position(2.35123456789, -1.123456789)
        before = waypoint.x, waypoint.y, waypoint.angle
        maps = (MapGeometry(0.05, (-10.317, -20.219, 0), 500, 500),
                MapGeometry(0.10, (-30.123, -40.567, 0), 1000, 1000))
        pixels = []
        for _ in range(100):
            for geometry in maps:
                waypoint.bind_map(geometry)
                pixels.append((waypoint.pixel_x, waypoint.pixel_y))
                self.assertEqual((waypoint.x, waypoint.y, waypoint.angle), before)
        self.assertNotEqual(pixels[0], pixels[1])

    def test_only_actual_move_recomputes_position(self):
        waypoint = Waypoint(10, 20)
        geometry = MapGeometry(0.10, (-1.231, -4.567, 0), 500, 500)
        waypoint.bind_map(geometry)
        before = waypoint.x, waypoint.y
        waypoint.set_angle(1.571123456789)
        waypoint.bind_map(geometry)
        waypoint.set_position(waypoint.pixel_x, waypoint.pixel_y)
        self.assertEqual((waypoint.x, waypoint.y), before)
        waypoint.set_position(30.25, 40.75)
        self.assertEqual((waypoint.x, waypoint.y), geometry.pixel_to_world(30.25, 40.75))

    def test_invalid_geometry_is_rejected(self):
        for resolution in (0, -0.1, float('nan'), float('inf'), None, True):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                MapGeometry(resolution, (0, 0, 0), 500, 500)
        for origin in ((0, 0), (0, 0, 0.01), (float('nan'), 0, 0)):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                MapGeometry(0.05, origin, 500, 500)

    def test_frame_policy(self):
        self.assertIsNone(check_frames('map', 'map'))
        self.assertIsNone(check_frames('camera_init', 'camera_init'))
        self.assertIsNotNone(check_frames(None, 'camera_init'))
        self.assertIsNotNone(check_frames('map', None))
        with self.assertRaisesRegex(ValueError, 'Frame mismatch'):
            check_frames('map', 'camera_init')

    def test_invalid_map_metadata(self):
        for field, value in (('negate', 2), ('mode', 'invalid'), ('free_thresh', -1),
                             ('occupied_thresh', float('nan')), ('frame_id', ''), ('image', 1)):
            data = {'resolution': 0.1, 'origin': [0, 0, 0], field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_map_metadata(data)


if __name__ == '__main__':
    unittest.main()
