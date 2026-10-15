"""Continuous image coordinates and world positions, independent of Qt.

The image boundary is (0, 0) at top left; a cell (column, row) has its
centre at (column + 0.5, row + 0.5). Waypoint drawing uses continuous
coordinates, so no half-cell offset or integer rounding is applied here.
Map origin is a map-local pose, never a transform between ROS frames.
"""

from dataclasses import dataclass
from math import isfinite


def finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def frame_id(value):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError('frame_id must be a non-empty string')
    return value


def check_frames(map_frame, waypoint_frame):
    """Reject known mismatches; return a warning when alignment is unknown."""
    frame_id(map_frame)
    frame_id(waypoint_frame)
    if map_frame is not None and waypoint_frame is not None:
        if map_frame != waypoint_frame:
            raise ValueError(
                f'Frame mismatch: map={map_frame}, waypoints={waypoint_frame}. '
                'TF conversion is not supported.')
        return None
    return ('地図またはWaypointのframe_idが不明です。座標系の一致は保証できません。'
            'TF変換は行いません。')


def validate_map_metadata(data):
    if not isinstance(data, dict):
        raise ValueError('Map YAML must be a mapping')
    resolution = finite_number(data.get('resolution'), 'resolution')
    if resolution <= 0:
        raise ValueError('resolution must be positive')
    origin = data.get('origin')
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise ValueError('origin must contain [x, y, yaw]')
    origin = tuple(finite_number(v, 'origin') for v in origin)
    if origin[2] != 0.0:
        raise ValueError('Non-zero map origin yaw is not supported')
    if 'image' in data and (not isinstance(data['image'], str) or not data['image']):
        raise ValueError('image must be a non-empty path')
    frame_id(data.get('frame_id'))
    for key, default in (('occupied_thresh', 0.65), ('free_thresh', 0.25)):
        if not 0 <= finite_number(data.get(key, default), key) <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if data.get('negate', 0) not in (0, 1):
        raise ValueError('negate must be 0 or 1')
    if data.get('mode', 'trinary') not in ('trinary', 'scale', 'raw'):
        raise ValueError('Unsupported map mode')
    return resolution, origin


@dataclass(frozen=True)
class MapGeometry:
    resolution: float
    origin: tuple
    width: int
    height: int

    def __post_init__(self):
        resolution, origin = validate_map_metadata(
            {'resolution': self.resolution, 'origin': self.origin})
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Map dimensions must be positive')
        object.__setattr__(self, 'resolution', resolution)
        object.__setattr__(self, 'origin', origin)

    def world_to_pixel(self, x, y):
        return ((x - self.origin[0]) / self.resolution,
                self.height - (y - self.origin[1]) / self.resolution)

    def pixel_to_world(self, x, y):
        return (self.origin[0] + x * self.resolution,
                self.origin[1] + (self.height - y) * self.resolution)


@dataclass(frozen=True)
class _PixelOriginGeometry:
    """Adapter for the older modular editor's world-zero pixel API."""

    resolution: float
    origin_x: float
    origin_y: float

    def world_to_pixel(self, x, y):
        return self.origin_x + x / self.resolution, self.origin_y - y / self.resolution

    def pixel_to_world(self, x, y):
        return (x - self.origin_x) * self.resolution, (self.origin_y - y) * self.resolution


class WorldPosition:
    """Keep world coordinates authoritative once a position has been set."""

    _geometry = None
    _has_world_position = False

    def set_world_position(self, x, y):
        self.x = finite_number(x, 'x')
        self.y = finite_number(y, 'y')
        self._has_world_position = True
        if self._geometry is not None:
            self.pixel_x, self.pixel_y = self._geometry.world_to_pixel(self.x, self.y)
        self.update_display_name()

    def bind_map(self, geometry):
        self._geometry = geometry
        self.resolution = geometry.resolution
        if self._has_world_position:
            self.pixel_x, self.pixel_y = geometry.world_to_pixel(self.x, self.y)
        else:
            self.set_world_position(*geometry.pixel_to_world(self.pixel_x, self.pixel_y))

    def update_metric_coordinates(self, origin_x, origin_y, resolution):
        """Compatibility adapter for callers passing world-zero in pixels.

        This legacy API explicitly synchronizes pixels to world. New code uses
        bind_map for projection and set_position for actual GUI movement.
        """
        resolution = finite_number(resolution, 'resolution')
        if resolution <= 0:
            raise ValueError('resolution must be positive')
        self._geometry = _PixelOriginGeometry(resolution, origin_x, origin_y)
        self.resolution = resolution
        self.x, self.y = self._geometry.pixel_to_world(self.pixel_x, self.pixel_y)
        self._has_world_position = True
        self.update_display_name()

    def set_position(self, x, y):
        if (x, y) == (self.pixel_x, self.pixel_y):
            return
        self.pixel_x, self.pixel_y = x, y
        if self._geometry is not None:
            self.set_world_position(*self._geometry.pixel_to_world(x, y))
