import numpy as np
import numpy.typing as npt
import pygfx as gfx
import tensorstore as ts
from funlib.geometry import Coordinate, Roi


class WrappingBuffer:
    """
    A buffer for volumetric data that operates in world coordinates.
    Handles chunking, wrapping, and efficient data movement into a gfx.Texture.
    """

    uniform_type = dict(
        current_logical_offset_in_pixels="3xi4",
        current_logical_shape_in_pixels="3xi4",
    )

    def __init__(
        self,
        backing_data: npt.NDArray,
        segmentations: npt.NDArray,
        shape_in_chunks: tuple[int, int, int] | Coordinate,
        chunk_shape_in_pixels: tuple[int, int, int] | Coordinate = None,
    ):
        """
        Args:
            backing_data (npt.NDArray):
                The source data (numpy or zarr).
            segmentations (npt.NDArray):
                The segmentation data (numpy or zarr).
            shape_in_chunks (tuple[int, int, int] or Coordinate):
                The shape of the wrapping buffer in chunks.
            chunk_shape_in_pixels (tuple[int, int, int] or Coordinate, optional):
                The shape of a chunk in pixels. If not provided, it will be inferred from the backing array.
        """
        self.backing_data = backing_data
        self.segmentations = segmentations
        self.shape_in_chunks = Coordinate(shape_in_chunks)
        self.chunk_shape_in_pixels = Coordinate(chunk_shape_in_pixels)
        self.shape_in_pixels = self.shape_in_chunks * self.chunk_shape_in_pixels

        # noinspection PyTypeChecker
        self.texture = gfx.Texture(
            data=np.zeros(self.shape_in_pixels, np.float32),
            dim=3,
        )

        # noinspection PyTypeChecker
        self.segmentations_texture = gfx.Texture(
            data=np.zeros(self.shape_in_pixels, np.uint32),
            dim=3,
        )

        # create our uniform buffer
        # we need to create this BEFORE we set _current_logical_roi_in_pixels
        self.uniform_buffer = gfx.Buffer(
            gfx.utils.array_from_shadertype(self.uniform_type), force_contiguous=True
        )
        # this will fill our uniform buffer with data
        self._current_logical_roi_in_pixels = None
        self._current_logical_roi_in_chunks: Roi | None = None

    @property
    def _current_logical_roi_in_pixels(self) -> Roi | None:
        # this could raise an AttributeError if not created yet
        return self.__current_logical_roi_in_pixels

    @_current_logical_roi_in_pixels.setter
    def _current_logical_roi_in_pixels(self, value: Roi | None):
        self.__current_logical_roi_in_pixels = value
        # indexing in the shader is done Fortran style (z, y, x), but these dimensions
        # all assume numpy/C style indexing (x, y, z). we pass the dimensions in Fortran
        # style to the shader so the shader completely operates in Fortran style.
        if value is not None:
            self.uniform_buffer.data["current_logical_offset_in_pixels"] = np.array(
                value.offset
            ).astype(int)[::-1]
            self.uniform_buffer.data["current_logical_shape_in_pixels"] = np.array(
                value.shape
            ).astype(int)[::-1]
        else:
            self.uniform_buffer.data["current_logical_offset_in_pixels"] = np.array(
                (0, 0, 0)
            ).astype(int)[::-1]
            self.uniform_buffer.data["current_logical_shape_in_pixels"] = np.array(
                (0, 0, 0)
            ).astype(int)[::-1]
        self.uniform_buffer.update_full()

    def get_snapped_roi_in_pixels(self, logical_roi_in_pixels: Roi) -> Roi:
        data_shape = self.backing_data.shape
        data_roi_shape_in_pixels = Roi((0, 0, 0), data_shape)

        intersected_roi_in_pixels = logical_roi_in_pixels.intersect(
            data_roi_shape_in_pixels
        )
        if intersected_roi_in_pixels.empty:
            return intersected_roi_in_pixels

        snapped_roi = intersected_roi_in_pixels.snap_to_grid(
            voxel_size=self.chunk_shape_in_pixels, mode="grow"
        )
        return snapped_roi

    def can_load_logical_roi(self, logical_roi_in_pixels: Roi) -> bool:
        """
        Snap the ROI to chunk bounds, then check if it fits in the buffer.
        Returns True if it fits, False otherwise.
        """
        # Snap to chunk bounds (grow)
        roi_shape = logical_roi_in_pixels.shape
        buffer_shape = self.shape_in_pixels
        # check each dimension of the Roi and ensure it fits within the buffer
        if any(roi_dim > buf_dim for roi_dim, buf_dim in zip(roi_shape, buffer_shape)):
            return False
        return True

    def load_logical_roi(self, logical_roi_in_pixels: Roi):
        snapped_roi = self.get_snapped_roi_in_pixels(logical_roi_in_pixels)
        if not self.can_load_logical_roi(logical_roi_in_pixels) or snapped_roi.empty:
            return
        logical_roi_in_chunks = snapped_roi / self.chunk_shape_in_pixels

        # snapped_roi and logical_roi_in_chunks now represent the final region to be loaded
        # we don't assign to the current_rois though because we need still need to do a comparison

        if self._current_logical_roi_in_chunks is None:
            to_load = [logical_roi_in_chunks]
        else:
            to_load = subtract_rois(
                logical_roi_in_chunks, self._current_logical_roi_in_chunks
            )

        # now we can update the current_rois
        self._current_logical_roi_in_pixels = snapped_roi
        self._current_logical_roi_in_chunks = logical_roi_in_chunks

        for logical_roi_in_chunks in to_load:
            for (
                buffer_roi_in_chunks,
                logical_roi_in_chunks,
            ) in self.wrap_logical_roi_into_buffer_rois(logical_roi_in_chunks):
                self.load_into_buffer(buffer_roi_in_chunks, logical_roi_in_chunks)

    def wrap_logical_roi_into_buffer_rois(
        self, logical_roi_in_chunks: Roi
    ) -> list[tuple[Roi, Roi]]:
        """
        Split a single logical Roi along buffer boundaries and convert it into a list of buffer Rois.

        Args:
            logical_roi_in_chunks (Roi):
                A logical Roi in chunk coordinates that is within the bounds of the backing data.
                This Roi may not be larger than the buffer Roi. However, it may cross buffer boundaries.

        Returns:
            A list of tuples (a, b) where a is the buffer Roi and b is the corresponding logical Roi. b will not
            cross any buffer boundaries.
        """
        # Note: since logical_roi_in_chunks cannot be larger than the buffer Roi, it can also only cross
        # buffer boundaries once in each dimension. If it were to cross twice, it would be larger than the
        # buffer
        assert logical_roi_in_chunks.shape.dims == self.shape_in_chunks.dims, (
            "ROI and buffer must have same number of dimensions"
        )
        for i in range(logical_roi_in_chunks.shape.dims):
            assert logical_roi_in_chunks.shape[i] <= self.shape_in_chunks[i], (
                f"Logical ROI shape {logical_roi_in_chunks.shape} cannot be larger than buffer shape {self.shape_in_chunks} in any dimension"
            )

        if logical_roi_in_chunks.empty:
            return []

        from itertools import product

        offset = logical_roi_in_chunks.offset
        shape = logical_roi_in_chunks.shape
        end = tuple(o + s for o, s in zip(offset, shape))
        grid_shape = tuple(self.shape_in_chunks)
        grid_offset = tuple(0 for _ in grid_shape)  # buffer always starts at 0

        # For each dimension, find the points to split at (start, optional interior boundary, end)
        split_coords = []
        for d in range(len(offset)):
            start = offset[d]
            stop = end[d]
            buffer = grid_shape[d]
            # Compute first boundary after start
            # The buffer is periodic, so boundaries are at multiples of buffer
            # Find the first boundary after start
            if buffer == 0:
                split_coords.append([start, stop])
                continue
            boundary = ((start - grid_offset[d]) // buffer + 1) * buffer + grid_offset[
                d
            ]
            if boundary < stop:
                split_coords.append([start, boundary, stop])
            else:
                split_coords.append([start, stop])

        # Now make all sub-ROIs from the split coordinates
        result = []
        for corner in product(*[range(len(s) - 1) for s in split_coords]):
            sub_offset = tuple(split_coords[d][i] for d, i in enumerate(corner))
            sub_end = tuple(split_coords[d][i + 1] for d, i in enumerate(corner))
            sub_shape = tuple(e - o for o, e in zip(sub_offset, sub_end))
            if any(s == 0 for s in sub_shape):
                continue
            logical_subroi = Roi(sub_offset, sub_shape)
            buffer_offset = tuple((o % s) for o, s in zip(sub_offset, grid_shape))
            buffer_roi = Roi(buffer_offset, sub_shape)
            result.append((buffer_roi, logical_subroi))
        return result

    def load_into_buffer(self, buffer_roi_in_chunks: Roi, logical_roi_in_chunks: Roi):
        """
        Load a section of the data into the buffer.

        Args:
            buffer_roi_in_chunks (Roi):
                A buffer Roi in chunk coordinates that is within the bounds of the buffer.
            logical_roi_in_chunks (Roi):
                A logical Roi in chunk coordinates corresponding to the buffer_roi_in_chunks.

                This Roi MUST intersect with the backing data, and it MAY be partially outside the bounds of the
                backing data. It MUST NOT cross any buffer boundaries and MUST NOT be larger than the buffer Roi.
        """
        # Convert both ROIs to pixel space
        buffer_roi_in_pixels = buffer_roi_in_chunks * self.chunk_shape_in_pixels
        logical_roi_in_pixels = logical_roi_in_chunks * self.chunk_shape_in_pixels

        # Helper to ensure all indices are ints
        def roi_to_slices(roi):
            return tuple(
                slice(int(o), int(o) + int(s)) for o, s in zip(roi.offset, roi.shape)
            )

        # Check for empty ROI
        if logical_roi_in_pixels.empty or buffer_roi_in_pixels.empty:
            return

        # Ensure we are only loading the portion within the backing data
        loadable_logical_roi_in_pixels = Roi(
            shape=self.backing_data.shape, offset=(0, 0, 0)
        ).intersect(logical_roi_in_pixels)
        if loadable_logical_roi_in_pixels.empty:
            return
        # Shrink our destination Roi to match the shape
        actual_buffer_roi_in_pixels = Roi(
            offset=buffer_roi_in_pixels.offset,
            shape=loadable_logical_roi_in_pixels.shape,
        )
        if isinstance(self.backing_data, ts.TensorStore):
            src_slices = roi_to_slices(
                loadable_logical_roi_in_pixels + Coordinate(self.backing_data.origin)
            )
        else:
            src_slices = roi_to_slices(loadable_logical_roi_in_pixels)
        dst_slices = roi_to_slices(actual_buffer_roi_in_pixels)

        # All the data here should be readable
        data = self.backing_data[src_slices]
        # we need to explicitly read the data if it's a tensorstore
        if isinstance(data, ts.TensorStore):
            data = data.read().result()
        segmentation_data = self.segmentations[src_slices]
        if isinstance(segmentation_data, ts.TensorStore):
            segmentation_data = segmentation_data.read().result()

        # Write to both textures
        self.texture.data[dst_slices] = np.array(data, dtype=np.float32)
        self.texture.update_range(
            actual_buffer_roi_in_pixels.offset, actual_buffer_roi_in_pixels.shape
        )

        self.segmentations_texture.data[dst_slices] = np.array(
            segmentation_data, dtype=np.uint32
        )
        self.segmentations_texture.update_range(
            actual_buffer_roi_in_pixels.offset, actual_buffer_roi_in_pixels.shape
        )


def set_dim(coord: Coordinate, dim: int, value) -> Coordinate:
    """Return a copy of coord with coord[dim] replaced by value."""
    return Coordinate(*coord[:dim], value, *coord[dim + 1 :])


def subtract_rois(roi_a: Roi, roi_b: Roi) -> list[Roi]:
    if roi_a.empty:
        return []
    if roi_b.empty or not roi_a.intersects(roi_b):
        return [roi_a]

    roi_b = roi_a.intersect(roi_b)
    result = []

    base_begin = roi_a.begin
    base_end = roi_a.end

    for d in range(roi_a.dims):
        a0, a1 = roi_a.begin[d], roi_a.end[d]
        b0, b1 = roi_b.begin[d], roi_b.end[d]

        # Slab before B
        if a0 < b0:
            slab_begin = base_begin
            slab_end = set_dim(base_end, d, b0)
            result.append(
                Roi(slab_begin, tuple(e - o for o, e in zip(slab_begin, slab_end)))
            )
            base_begin = set_dim(base_begin, d, b0)

        # Slab after B
        if b1 < a1:
            slab_begin = set_dim(base_begin, d, b1)
            slab_end = base_end
            result.append(
                Roi(slab_begin, tuple(e - o for o, e in zip(slab_begin, slab_end)))
            )
            base_end = set_dim(base_end, d, b1)

    return result
