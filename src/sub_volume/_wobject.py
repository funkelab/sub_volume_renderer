import numpy as np
import numpy.typing as npt
import pygfx as gfx
import zarr
from funlib.geometry import Coordinate, Roi
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
        data_segmentation_pairs: list[
            tuple[npt.NDArray | zarr.Array, npt.NDArray | zarr.Array]
        ],
        buffer_shape_in_chunks: list[tuple[int, int, int]],
        chunk_shape_in_pixels: list[tuple[int, int, int]] | None = None,
    ):
        # Use the first (highest resolution) data for base volume dimensions
        base_data = data_segmentation_pairs[0][0]
        self.volume_dimensions = base_data.shape
        num_scales = len(data_segmentation_pairs)

        # Handle per-scale or uniform buffer configurations
        if isinstance(buffer_shape_in_chunks, tuple):
            # Single configuration for all scales
            buffer_shapes = [buffer_shape_in_chunks] * num_scales
        else:
            # Per-scale configurations
            buffer_shapes = buffer_shape_in_chunks
            if len(buffer_shapes) != num_scales:
                raise ValueError(
                    f"buffer_shape_in_chunks list length ({len(buffer_shapes)}) must match number of scales ({num_scales})"
                )

        # Handle per-scale or uniform chunk configurations
        if chunk_shape_in_pixels is None:
            # Try to infer from first data array
            if hasattr(base_data, "chunks"):
                chunk_shapes = [base_data.chunks] * num_scales
            else:
                raise ValueError(
                    "if chunk_shape_in_pixels is not provided, base data must have a 'chunks' attribute"
                )
        elif isinstance(chunk_shape_in_pixels, tuple):
            # Single configuration for all scales
            chunk_shapes = [chunk_shape_in_pixels] * num_scales
        else:
            # Per-scale configurations
            chunk_shapes = chunk_shape_in_pixels
            if len(chunk_shapes) != num_scales:
                raise ValueError(
                    f"chunk_shape_in_pixels list length ({len(chunk_shapes)}) must match number of scales ({num_scales})"
                )

        # Validate chunk shapes match data dimensions
        for i, (scale_data, _) in enumerate(data_segmentation_pairs):
            if len(chunk_shapes[i]) != scale_data.ndim:
                raise ValueError(
                    f"chunk_shape_in_pixels[{i}] length must match data dimensions"
                )

        # Create multiple WrappingBuffers for each scale level
        self.wrapping_buffers = []
        for i, (scale_data, scale_segmentations) in enumerate(data_segmentation_pairs):
            # Calculate scale factor for same-sized voxels
            # For voxels to appear same size: lower resolution needs smaller coordinate scaling
            # Scale 0 (full res): scale_factor = 1.0
            # Scale 1 (half res): scale_factor = 0.5 (sample at half coordinates)
            scale_factor = tuple(
                float(scale_data.shape[j]) / float(base_data.shape[j]) for j in range(3)
            )

            buffer = WrappingBuffer(
                backing_data=scale_data,
                segmentations=scale_segmentations,
                shape_in_chunks=buffer_shapes[i],
                chunk_shape_in_pixels=chunk_shapes[i],
                scale_factor=scale_factor,
            )
            self.wrapping_buffers.append(buffer)

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
    def textures(self) -> list[gfx.Texture]:
        """Return all scale level textures."""
        return [buffer.texture for buffer in self.wrapping_buffers]

    @property
    def segmentations_textures(self) -> list[gfx.Texture]:
        """Return all scale level segmentations textures."""
        return [buffer.segmentations_texture for buffer in self.wrapping_buffers]

    def center_on_position(
        self,
        position: tuple[float, float, float],
        sizes: list[tuple[int, int, int]] | None = None,
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
        if sizes is None:
            sizes = [
                (wrapping_buffer.shape_in_chunks - Coordinate(1, 1, 1))
                * wrapping_buffer.chunk_shape_in_pixels
                for wrapping_buffer in self.wrapping_buffers
            ]
        if len(sizes) != len(self.wrapping_buffers):
            raise ValueError(
                f"sizes list length ({len(sizes)}) must match number of scales ({len(self.wrapping_buffers)})"
            )

        # convert the world position to our local space using the inverse world matrix
        # we need to attach and then remove the homogeneous coordinate to play nice with the matrix multiplication
        camera_data_pos = tuple(self.world.inverse_matrix @ np.array([*position, 1]))[
            :3
        ]
        camera_data_pos = camera_data_pos[::-1]
        logical_rois = [
            Roi(
                tuple(
                    int(c * f - s // 2)
                    for c, s, f in zip(camera_data_pos, size, buffer.scale_factor[::-1])
                ),
                size,
            )
            for size, buffer in zip(sizes, self.wrapping_buffers)
        ]
        # Update all wrapping buffers with scale-appropriate ROIs
        for buffer, logical_roi in zip(self.wrapping_buffers, logical_rois):
            if buffer.can_load_logical_roi(logical_roi):
                buffer.load_logical_roi(logical_roi)

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
