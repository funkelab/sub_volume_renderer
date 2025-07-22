import numpy as np
import numpy.typing as npt
import pygfx as gfx
import zarr
from funlib.geometry import Roi
from pygfx import WorldObject
from pygfx.utils.bounds import Bounds

from ._material import SubVolumeMaterial
from ._wrapping_buffer import WrappingBuffer


class SubVolume(gfx.Volume):
    uniform_type = dict(
        WorldObject.uniform_type,
        volume_dimensions="3xf4",
    )
    material: SubVolumeMaterial

    def __init__(
        self,
        material: SubVolumeMaterial,
        data: npt.NDArray | zarr.Array,
        segmentations: npt.NDArray | zarr.Array,
        buffer_shape_in_chunks: tuple[int, int, int],
        chunk_shape_in_pixels: tuple[int, int, int] | None = None,
    ):
        self.volume_dimensions = data.shape

        if chunk_shape_in_pixels is None and hasattr(data, "chunks"):
            chunk_shape_in_pixels = data.chunks
        elif chunk_shape_in_pixels is None:
            raise ValueError(
                "if chunk_shape_in_pixels is not provided, data must have a 'chunks' attribute"
            )
        assert len(chunk_shape_in_pixels) == data.ndim

        self.wrapping_buffer = WrappingBuffer(
            backing_data=data,
            segmentations=segmentations,
            shape_in_chunks=buffer_shape_in_chunks,
            chunk_shape_in_pixels=chunk_shape_in_pixels,
        )

        geometry = gfx.box_geometry(*self.volume_dimensions)
        super().__init__(
            geometry=geometry,
            material=material,
        )

        # indexing in the shader is done Fortran style (z, y, x), but these dimensions
        # all assume numpy/C style indexing (x, y, z). we pass the dimensions in Fortran
        # style to the shader so the shader completely operates in Fortran style.
        self.uniform_buffer.data["volume_dimensions"] = np.array(
            tuple(self.volume_dimensions)[::-1], dtype=np.float32
        )

    @property
    def texture(self) -> gfx.Texture:
        return self.wrapping_buffer.texture

    @property
    def segmentations_texture(self) -> gfx.Texture:
        return self.wrapping_buffer.segmentations_texture

    def center_on_position(
        self,
        position: tuple[float, float, float],
        size: tuple[int, int, int] | None = None,
    ):
        """
        Center the sub volume on a given position in world coordinates.

        Args:
            position (tuple[float, float, float]):
                The world position to center the sub volume on, as a tuple of (x, y, z).
            size (tuple[int, int, int] | None):
                The size of the sub volume to load, as a tuple of (width, height, depth).
                 If not passed, the size will be chosen to max out the available space in the wrapping buffer.
        """
        if size is None:
            size = self.wrapping_buffer.shape_in_pixels

        # convert the world position to our local space using the inverse world matrix
        # we need to attach and then remove the homogeneous coordinate to play nice with the matrix multiplication
        camera_data_pos = tuple(self.world.inverse_matrix @ np.array([*position, 1]))[
            :3
        ]
        camera_data_pos = camera_data_pos[::-1]
        logical_roi = Roi(
            tuple(int(c - s // 2) for c, s in zip(camera_data_pos, size)),
            size,
        )
        # do we really need this check if load_logical_roi does the check internally?
        # inclined to keep since the internal check is an implementation detail
        if self.wrapping_buffer.can_load_logical_roi(logical_roi):
            self.wrapping_buffer.load_logical_roi(logical_roi)

    def _get_bounds_from_geometry(self):
        if self._bounds_geometry is not None:
            return self._bounds_geometry
        # account for multi-channel image data
        grid_shape = tuple(reversed(self.volume_dimensions))
        # create aabb in index/data space
        aabb = np.array([np.zeros_like(grid_shape), grid_shape[::-1]], dtype="f8")
        # convert to local image space by aligning
        # center of voxel index (0, 0, 0) with origin (0, 0, 0)
        aabb -= 0.5
        # ensure coordinates are 3D
        # NOTE: important we do this last, we don't want to apply
        # the -0.5 offset to the z-coordinate of 2D images
        if aabb.shape[1] == 2:
            aabb = np.hstack([aabb, [[0], [0]]])
        self._bounds_geometry = Bounds(aabb, None)
        return self._bounds_geometry
