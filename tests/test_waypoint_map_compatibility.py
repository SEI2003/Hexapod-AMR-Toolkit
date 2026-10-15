"""Exercise the normal entry point with Qt offscreen, without an event loop."""

import os
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import yaml
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

import main_all

_APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]


def document(version='1.1', frame='map'):
    data = {
        'format_version': version,
        'waypoints': [
            {'number': 7, 'x': 2.35123456789123, 'y': -1.12345678912345,
             'angle_radians': 0.785123456789, 'stop': False, 'wait_time': 0.0,
             'manual': False, 'change_map': '', 'extra': {'terrain': [0, 17, 99]}},
            {'number': 42, 'x': 5.40123456789123, 'y': 3.21123456789123,
             'angle_radians': 1.571123456789, 'threshold': -1.0, 'rotate': 0.0},
        ],
    }
    if frame is not None:
        data['frame_id'] = frame
    return data


class TestWaypointMapCompatibility(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ)
        self.env_patch.start()
        os.environ.pop('AMR_TOOLKIT_MAP_FRAME', None)
        # Each offscreen window owns its format observers for this test only.
        manager_patch = patch.object(main_all, 'format_manager', main_all.FormatManager())
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        self.window = main_all.MainWindow()
        self.viewer = self.window.image_viewer
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.window.close)
        self.addCleanup(self.env_patch.stop)

    def load_map(self, resolution=0.05, origin=(-10.317, -20.219, 0),
                 size=(500, 500), frame='map'):
        width, height = size
        self.viewer.load_image(np.full((height, width), 254, np.uint8), width, height)
        data = {'resolution': resolution, 'origin': list(origin)}
        if frame is not None:
            data['frame_id'] = frame
        self.viewer.apply_map_metadata(data)

    def write_map(self, name, resolution=0.1, origin=(-2.317, -3.219, 0), frame='map'):
        folder = Path(self.temp.name)
        pgm = folder / f'{name} image.pgm'
        pixels = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
        pgm.write_bytes(b'P5\n# terrain classes\n256 64\n255\n' + pixels.tobytes())
        data = {'image': pgm.name, 'resolution': resolution, 'origin': list(origin),
                'negate': 1, 'occupied_thresh': 0.78, 'free_thresh': 0.12, 'mode': 'raw'}
        if frame is not None:
            data['frame_id'] = frame
        path = folder / f'{name}.yaml'
        path.write_text(yaml.safe_dump(data))
        return path, pgm, pixels, data

    def test_yaml_file_roundtrips_after_display_and_map_switches(self):
        for version in ('1.0', '1.1'):
            for frame in ('map', 'camera_init', None):
                with self.subTest(version=version, frame=frame):
                    # Import before a map exists, then display on both geometries.
                    self.viewer.load_image(np.zeros((2, 2), np.uint8), 2, 2)
                    data = document(version, frame)
                    path = Path(self.temp.name) / 'waypoints.yaml'
                    path.write_text(yaml.safe_dump(data))
                    self.window.import_waypoints_yaml(str(path))
                    for _ in range(3):
                        self.load_map(frame=frame)
                        first_pixels = [(wp.pixel_x, wp.pixel_y) for wp in self.viewer.waypoints]
                        self.load_map(0.1, (-30.123, -40.567, 0), (1000, 1000), frame)
                        self.assertNotEqual(first_pixels,
                                            [(wp.pixel_x, wp.pixel_y) for wp in self.viewer.waypoints])
                        self.viewer.render_composite_pixmap()
                        self.assertEqual(self.window.export_waypoints_data(), data)
                    with patch.object(main_all.QFileDialog, 'getSaveFileName', return_value=(str(path), '')):
                        self.window.export_waypoints_yaml()
                    self.assertEqual(yaml.safe_load(path.read_text()), data)
                    self.window.import_waypoints_yaml(str(path))
                    self.assertEqual(self.window.export_waypoints_data(), data)
                    self.assertEqual(set(self.window.right_panel.waypoint_widgets), {7, 42})

    def test_origin_and_size_changes_only_display_coordinates(self):
        self.viewer.import_waypoints_from_yaml(document())
        before = self.window.export_waypoints_data()
        self.load_map(0.05, (-10, -20, 0), (500, 500))
        wp = self.viewer.waypoints[0]
        first = wp.pixel_x, wp.pixel_y
        self.load_map(0.05, (-11, -18, 0), (1000, 1000))
        self.assertAlmostEqual(wp.pixel_x - first[0], 20)
        self.assertAlmostEqual(wp.pixel_y - first[1], 540)
        self.assertEqual(self.window.export_waypoints_data(), before)

    def test_off_map_positions_with_small_positive_resolution_can_be_saved(self):
        self.viewer.import_waypoints_from_yaml(document())
        before = self.window.export_waypoints_data()
        self.load_map(1e-9, (-100, -100, 0), (500, 500))
        self.assertIsNotNone(self.viewer.render_composite_pixmap())
        self.assertEqual(self.window.export_waypoints_data(), before)

    def test_frame_mismatch_rejects_import_atomically(self):
        self.load_map(frame='map')
        self.viewer.import_waypoints_from_yaml(document())
        before = self.window.export_waypoints_data()
        with self.assertRaisesRegex(ValueError, 'Frame mismatch'):
            self.viewer.import_waypoints_from_yaml(document(frame='camera_init'))
        self.assertEqual(self.window.export_waypoints_data(), before)

    def test_unknown_frames_warn_and_are_never_invented(self):
        self.load_map(frame=None)
        with patch.object(self.viewer, 'show_edit_message') as warning:
            self.viewer.import_waypoints_from_yaml(document(frame='camera_init'))
        warning.assert_called_once()
        self.assertEqual(self.window.export_waypoints_data()['frame_id'], 'camera_init')
        self.viewer.import_waypoints_from_yaml(document(frame=None))
        self.assertNotIn('frame_id', self.window.export_waypoints_data())

    def test_explicit_map_frame_setting_and_atomic_rejection(self):
        path, _, _, _ = self.write_map('unknown', frame=None)
        self.viewer.import_waypoints_from_yaml(document(frame='camera_init'))
        os.environ['AMR_TOOLKIT_MAP_FRAME'] = 'camera_init'
        self.window.load_yaml_file(str(path))
        self.assertEqual(self.viewer.map_frame_id, 'camera_init')
        before = self.viewer.map_geometry
        os.environ['AMR_TOOLKIT_MAP_FRAME'] = 'map'
        with patch.object(main_all.QMessageBox, 'warning') as error:
            self.window.load_yaml_file(str(path))
        error.assert_called_once()
        self.assertEqual(self.viewer.map_geometry, before)

    def test_invalid_map_and_missing_image_leave_loaded_state_intact(self):
        self.load_map()
        self.viewer.import_waypoints_from_yaml(document())
        before_geometry = self.viewer.map_geometry
        before_data = self.window.export_waypoints_data()
        for name, resolution, origin in (('yaw', 0.1, (0, 0, 0.1)),
                                          ('resolution', 0, (0, 0, 0)),
                                          ('missing', 0.1, (0, 0, 0))):
            path, pgm, _, _ = self.write_map(name, resolution, origin)
            if name == 'missing':
                pgm.unlink()
            with patch.object(main_all.QMessageBox, 'warning') as error:
                self.window.load_yaml_file(str(path))
            error.assert_called_once()
            self.assertEqual(self.viewer.map_geometry, before_geometry)
            self.assertEqual(self.window.export_waypoints_data(), before_data)

    def test_known_frame_mismatch_rejects_map_before_image_changes(self):
        self.load_map()
        self.viewer.import_waypoints_from_yaml(document())
        original_pixels = self.viewer.map_image_array.copy()
        path, _, _, _ = self.write_map('camera', frame='camera_init')
        with patch.object(main_all.QMessageBox, 'warning') as error:
            self.window.load_yaml_file(str(path))
        error.assert_called_once()
        np.testing.assert_array_equal(self.viewer.map_image_array, original_pixels)
        self.assertEqual(self.viewer.map_frame_id, 'map')

    def test_terrain_pixels_and_map_metadata_preserved_on_explicit_export(self):
        path, pgm, pixels, metadata = self.write_map('terrain')
        original_pgm, original_yaml = pgm.read_bytes(), path.read_bytes()
        self.window.load_yaml_file(str(path))
        self.assertEqual(self.viewer.map_metadata, metadata)
        np.testing.assert_array_equal(self.viewer.map_image_array, pixels)
        self.viewer.import_waypoints_from_yaml(document())
        self.viewer.waypoints[0].set_angle(2.0)
        self.assertEqual(pgm.read_bytes(), original_pgm)
        self.assertEqual(path.read_bytes(), original_yaml)
        exported = Path(self.temp.name) / 'export.pgm'
        with patch.object(main_all.QFileDialog, 'getSaveFileName', return_value=(str(exported), '')):
            self.window.export_pgm_with_drawings()
        array, _, _ = main_all.read_pgm_file(str(exported))
        np.testing.assert_array_equal(array, pixels)
        expected = dict(metadata, image='export.pgm')
        self.assertEqual(yaml.safe_load(exported.with_suffix('.yaml').read_text()), expected)

    def test_bare_pgm_does_not_reuse_old_geometry(self):
        self.load_map()
        self.viewer.import_waypoints_from_yaml(document())
        before = self.window.export_waypoints_data()
        _, pgm, _, _ = self.write_map('bare')  # Its YAML has a different stem.
        self.window.load_pgm_file(str(pgm))
        self.assertIsNone(self.viewer.map_geometry)
        self.assertIsNone(self.viewer.origin_point)
        self.assertIsNone(self.viewer.waypoints[0]._geometry)
        self.viewer.add_waypoint(QPoint(1, 1))
        self.assertEqual(self.window.export_waypoints_data(), before)

    def test_pgm_entry_uses_sibling_yaml(self):
        path, pgm, _, _ = self.write_map('sibling')
        sibling_pgm = path.with_suffix('.pgm')
        pgm.rename(sibling_pgm)
        data = yaml.safe_load(path.read_text())
        data['image'] = sibling_pgm.name
        path.write_text(yaml.safe_dump(data))
        self.window.load_pgm_file(str(sibling_pgm))
        self.assertEqual(self.viewer.resolution, 0.1)
        self.assertEqual(self.viewer.map_geometry.origin, (-2.317, -3.219, 0))

    def test_existing_repository_yaml_formats(self):
        data = yaml.safe_load((ROOT / 'map/atc/atc_sample_waypoint.yaml').read_text())
        self.viewer.import_waypoints_from_yaml(data)
        path = ROOT / 'map/atc/ubuntu_export_test/test.yaml'
        self.window.load_yaml_file(str(path))
        self.assertEqual(self.viewer.resolution, 0.05)
        self.assertEqual(self.window.export_waypoints_data(), data)

    def test_legacy_missing_version_number_and_frame(self):
        self.load_map()
        self.viewer.import_waypoints_from_yaml({'waypoints': [{'x': 1.23456789, 'y': 2.0,
                                                               'angle_radians': 0.1}]})
        exported = self.window.export_waypoints_data()
        self.assertEqual(exported['format_version'], '1.0')
        self.assertNotIn('frame_id', exported)
        self.assertEqual(exported['waypoints'][0]['number'], 1)
        self.assertEqual(exported['waypoints'][0]['x'], 1.23456789)

    def test_invalid_waypoints_do_not_replace_existing_document(self):
        self.viewer.import_waypoints_from_yaml(document())
        before = self.window.export_waypoints_data()
        for invalid in ({'x': float('nan')}, {'number': 0}, {'angle_radians': None},
                        {'number': 42}):
            data = document()
            data['waypoints'][0].update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.viewer.import_waypoints_from_yaml(data)
            self.assertEqual(self.window.export_waypoints_data(), before)

    def test_custom_format_import_export(self):
        custom = {'version': 'robot-custom', 'format': {
            'number': 'int', 'x': 'float', 'y': 'float', 'angle_radians': 'float',
            'speed': 'float', 'enabled': 'bool'}}
        main_all.format_manager.set_format(custom)
        data = {'format_version': 'robot-custom', 'frame_id': 'camera_init',
                'route_name': 'stairs', 'waypoints': [
                    {'number': 9, 'x': 2.123456789, 'y': -3.123456789,
                     'angle_radians': 0.123456789, 'speed': 0.25, 'enabled': True,
                     'extension': {'gait': 'tripod'}}]}
        self.viewer.import_waypoints_from_yaml(data)
        self.assertEqual(self.window.export_waypoints_data(), data)
        custom = deepcopy(custom)
        custom['version'] = 'robot-v2'
        del custom['format']['enabled']
        main_all.format_manager.set_format(custom)
        exported = self.window.export_waypoints_data()
        self.assertEqual(exported['format_version'], 'robot-v2')
        self.assertNotIn('enabled', exported['waypoints'][0])

    def test_existing_attribute_editor_keeps_unknown_extension_fields(self):
        self.viewer.import_waypoints_from_yaml(document())
        wp = self.viewer.waypoints[0]
        dialog = main_all.AttributeDialog(wp, main_all.format_manager.get_format())
        try:
            self.assertEqual(dialog.get_attributes()['extra'], {'terrain': [0, 17, 99]})
            self.assertEqual((wp.x, wp.y, wp.angle),
                             (document()['waypoints'][0]['x'], document()['waypoints'][0]['y'],
                              document()['waypoints'][0]['angle_radians']))
        finally:
            dialog.close()

    def test_add_drag_angle_reorder_delete_and_history(self):
        self.load_map(0.1)
        label = self.viewer.pgm_display
        with patch.object(self.viewer, 'display_to_image_coords', side_effect=lambda p: p):
            self.viewer.add_waypoint(QPoint(100, 100))
            wp = self.viewer.waypoints[0]
            self.assertEqual((wp.x, wp.y), self.viewer.map_geometry.pixel_to_world(100, 100))
            self.assertEqual(self.window.export_waypoints_data()['frame_id'], 'map')
            self.viewer.enter_edit_mode(wp)

            def mouse(kind, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
                button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
                event = QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button,
                                    Qt.MouseButton.LeftButton, modifiers)
                {QEvent.Type.MouseButtonPress: label.mousePressEvent,
                 QEvent.Type.MouseMove: label.mouseMoveEvent,
                 QEvent.Type.MouseButtonRelease: label.mouseReleaseEvent}[kind](event)

            mouse(QEvent.Type.MouseButtonPress, 100, 100)
            mouse(QEvent.Type.MouseMove, 101, 101)  # Still inside its tooltip area.
            mouse(QEvent.Type.MouseMove, 130, 120)
            mouse(QEvent.Type.MouseButtonRelease, 130, 120)
            self.assertEqual((wp.x, wp.y), self.viewer.map_geometry.pixel_to_world(130, 120))
            before = wp.x, wp.y
            mouse(QEvent.Type.MouseButtonPress, 130, 120, Qt.KeyboardModifier.ShiftModifier)
            mouse(QEvent.Type.MouseMove, 130, 100, Qt.KeyboardModifier.ShiftModifier)
            mouse(QEvent.Type.MouseButtonRelease, 130, 100)
            self.assertEqual((wp.x, wp.y), before)
            self.assertAlmostEqual(wp.angle, np.pi / 2)
            self.viewer.undo()
            self.assertEqual((wp.x, wp.y), before)
            self.assertEqual(wp.angle, 0)
            self.viewer.redo()
            self.assertAlmostEqual(wp.angle, np.pi / 2)
            self.viewer.add_waypoint(QPoint(200, 200))
        second = self.viewer.waypoints[1]
        second_position = second.x, second.y
        self.viewer.reorder_waypoints(second.number, wp.number)
        self.assertIs(self.viewer.waypoints[0], second)
        self.assertEqual((second.x, second.y), second_position)
        self.assertEqual([p.number for p in self.viewer.waypoints], [1, 2])
        self.viewer.remove_waypoint(wp.number)
        self.assertEqual(self.viewer.waypoints, [second])
        self.assertEqual((second.x, second.y), second_position)

    def test_angle_and_click_without_moving_keep_imported_precision(self):
        self.load_map(0.1)
        self.viewer.import_waypoints_from_yaml(document())
        wp = self.viewer.waypoints[0]
        before = wp.x, wp.y
        wp.set_position(wp.pixel_x, wp.pixel_y)
        wp.set_angle(0.123456789)
        self.viewer.handle_waypoint_edited(wp)
        self.assertEqual((wp.x, wp.y), before)
        self.load_map(0.05, (-5.123, -8.456, 0), (1000, 500))
        self.viewer.undo()
        self.assertEqual((wp.x, wp.y), before)
        self.assertEqual(wp.angle, document()['waypoints'][0]['angle_radians'])
        self.assertEqual((wp.pixel_x, wp.pixel_y), self.viewer.map_geometry.world_to_pixel(*before))

    def test_stationary_click_and_hover_do_not_change_imported_position(self):
        self.load_map(0.1)
        self.viewer.import_waypoints_from_yaml(document())
        wp = self.viewer.waypoints[0]
        before = self.window.export_waypoints_data()
        self.viewer.enter_edit_mode(wp)
        label = self.viewer.pgm_display
        position = QPointF(round(wp.pixel_x), round(wp.pixel_y))
        with patch.object(self.viewer, 'display_to_image_coords', side_effect=lambda p: p):
            press = QMouseEvent(QEvent.Type.MouseButtonPress, position, position,
                                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                Qt.KeyboardModifier.NoModifier)
            label.mousePressEvent(press)
            for buttons, pos in ((Qt.MouseButton.LeftButton, position),
                                 (Qt.MouseButton.NoButton, position + QPointF(10, 10))):
                move = QMouseEvent(QEvent.Type.MouseMove, pos, pos,
                                   Qt.MouseButton.NoButton, buttons, Qt.KeyboardModifier.NoModifier)
                label.mouseMoveEvent(move)
        self.assertEqual(self.window.export_waypoints_data(), before)

    def test_new_unknown_frame_is_omitted_and_imported_numbers_survive_addition(self):
        self.load_map(frame=None)
        with patch.object(self.viewer, 'display_to_image_coords', side_effect=lambda p: p):
            self.viewer.add_waypoint(QPoint(100, 100))
        self.assertNotIn('frame_id', self.window.export_waypoints_data())
        self.viewer.import_waypoints_from_yaml(document(frame=None))
        before = self.window.export_waypoints_data()['waypoints']
        with patch.object(self.viewer, 'display_to_image_coords', side_effect=lambda p: p):
            self.viewer.add_waypoint(QPoint(200, 200))
        exported = self.window.export_waypoints_data()
        self.assertEqual(exported['waypoints'][:2], before)
        self.assertEqual(exported['waypoints'][2]['number'], 43)
        self.assertNotIn('frame_id', exported)

    def test_mouse_metric_display_uses_resolution(self):
        self.load_map(0.1, (-2.317, -3.219, 0))
        with patch.object(self.viewer, 'display_to_image_coords', return_value=QPoint(100, 200)):
            self.viewer.update_mouse_position(QPoint(100, 200))
        self.assertIn('Metric: (7.68, 26.78)', self.viewer.coord_label.text())


if __name__ == '__main__':
    unittest.main()
