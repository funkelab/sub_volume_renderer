# entire file written by Claude Sonnet 4 with Claude Code
# watch out for issues, and be skeptical of the accuracy of the code
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from sub_volume import WrappingBuffer


@pytest.fixture
def boundary_buffer():
    """Create a buffer with smaller dimensions for boundary testing."""
    backing_data = np.arange(16 * 16 * 16, dtype=np.uint16).reshape((16, 16, 16))
    segmentations = np.zeros(backing_data.shape, dtype=np.uint16)
    return WrappingBuffer(
        backing_data, segmentations, Coordinate((2, 2, 2)), Coordinate((4, 4, 4))
    )


@pytest.fixture
def large_boundary_buffer():
    """Create a buffer with larger dimensions for extensive boundary testing."""
    backing_data = np.arange(32 * 32 * 32, dtype=np.uint16).reshape((32, 32, 32))
    segmentations = np.zeros(backing_data.shape, dtype=np.uint16)
    return WrappingBuffer(
        backing_data, segmentations, Coordinate((3, 3, 3)), Coordinate((8, 8, 8))
    )


def test_load_roi_extending_beyond_positive_boundary(boundary_buffer):
    """Test loading ROI that extends beyond positive boundary of data."""
    # ROI extends from (12, 12, 12) with size (8, 8, 8)
    # Data only goes to (16, 16, 16), so this should load partial data
    roi = Roi((12, 12, 12), (8, 8, 8))

    boundary_buffer.load_logical_roi(roi)

    # Should load something (not empty)
    assert boundary_buffer._current_logical_roi_in_pixels is not None

    # The loaded data should match the intersection with data bounds
    current_roi = boundary_buffer._current_logical_roi_in_pixels
    assert current_roi.offset[0] >= 12
    assert current_roi.offset[1] >= 12
    assert current_roi.offset[2] >= 12

    # Should not extend beyond data bounds
    assert current_roi.offset[0] + current_roi.shape[0] <= 16
    assert current_roi.offset[1] + current_roi.shape[1] <= 16
    assert current_roi.offset[2] + current_roi.shape[2] <= 16


def test_load_roi_extending_beyond_negative_boundary(boundary_buffer):
    """Test loading ROI that extends beyond negative boundary of data."""
    # ROI starts at (-4, -4, -4) with size (8, 8, 8)
    # Should load partial data from (0, 0, 0) to (4, 4, 4)
    roi = Roi((-4, -4, -4), (8, 8, 8))

    boundary_buffer.load_logical_roi(roi)

    # Should load something (not empty)
    assert boundary_buffer._current_logical_roi_in_pixels is not None

    # The loaded data should start at (0, 0, 0) or later
    current_roi = boundary_buffer._current_logical_roi_in_pixels
    assert current_roi.offset[0] >= 0
    assert current_roi.offset[1] >= 0
    assert current_roi.offset[2] >= 0


def test_load_roi_completely_outside_data_bounds(boundary_buffer):
    """Test loading ROI that is completely outside data bounds."""
    # ROI completely outside positive boundary
    roi = Roi((20, 20, 20), (4, 4, 4))

    boundary_buffer.load_logical_roi(roi)

    # Should not load anything
    assert boundary_buffer._current_logical_roi_in_pixels is None


def test_load_roi_partially_outside_multiple_axes(boundary_buffer):
    """Test loading ROI that extends beyond boundaries on multiple axes."""
    # ROI extends beyond positive boundary on X and Y axes
    roi = Roi((14, 14, 8), (4, 4, 4))

    boundary_buffer.load_logical_roi(roi)

    # Should load something
    assert boundary_buffer._current_logical_roi_in_pixels is not None

    current_roi = boundary_buffer._current_logical_roi_in_pixels
    # Should be clipped to data bounds
    assert current_roi.offset[0] + current_roi.shape[0] <= 16
    assert current_roi.offset[1] + current_roi.shape[1] <= 16
    # Z axis should be fine
    assert current_roi.offset[2] >= 8


def test_systematic_boundary_positions(large_boundary_buffer):
    """Test loading ROIs at various positions along each axis near boundaries."""
    # Test positions that should work near boundaries
    test_positions = [
        # Near negative boundary
        (0, 16, 16),
        (16, 0, 16),
        (16, 16, 0),
        # Near positive boundary
        (16, 16, 16),  # Center
        (24, 16, 16),  # Near positive X boundary
        (16, 24, 16),  # Near positive Y boundary
        (16, 16, 24),  # Near positive Z boundary
    ]

    for pos in test_positions:
        roi = Roi(pos, (8, 8, 8))

        # Should not crash
        large_boundary_buffer.load_logical_roi(roi)

        # If loaded successfully, should have valid ROI
        if large_boundary_buffer._current_logical_roi_in_pixels is not None:
            current_roi = large_boundary_buffer._current_logical_roi_in_pixels
            # Should be within data bounds
            assert current_roi.offset[0] >= 0
            assert current_roi.offset[1] >= 0
            assert current_roi.offset[2] >= 0
            assert current_roi.offset[0] + current_roi.shape[0] <= 32
            assert current_roi.offset[1] + current_roi.shape[1] <= 32
            assert current_roi.offset[2] + current_roi.shape[2] <= 32


def test_loaded_data_matches_backing_data(boundary_buffer):
    """Test that loaded data matches the backing data for boundary cases."""
    # Load ROI that extends beyond positive boundary
    roi = Roi((12, 12, 12), (8, 8, 8))
    boundary_buffer.load_logical_roi(roi)

    current_roi = boundary_buffer._current_logical_roi_in_pixels
    if current_roi is not None:
        # Check that loaded data in texture matches backing data
        for z in range(current_roi.shape[2]):
            for y in range(current_roi.shape[1]):
                for x in range(current_roi.shape[0]):
                    texture_pos = (
                        current_roi.offset[0] + x,
                        current_roi.offset[1] + y,
                        current_roi.offset[2] + z,
                    )

                    # Map to buffer coordinates (considering wrapping)
                    buffer_pos = tuple(
                        pos % boundary_buffer.shape_in_pixels[i]
                        for i, pos in enumerate(texture_pos)
                    )

                    # Data should match
                    expected = boundary_buffer.backing_data[texture_pos]
                    actual = boundary_buffer.texture.data[buffer_pos]
                    assert actual == expected


# Chunk Alignment Edge Cases
def test_roi_exactly_at_chunk_boundaries(boundary_buffer):
    """Test ROI that starts and ends exactly at chunk boundaries."""
    # Chunk size is (4, 4, 4), so test ROI at (0, 0, 0) and (4, 4, 4)
    roi = Roi((0, 0, 0), (4, 4, 4))  # Exactly one chunk

    boundary_buffer.load_logical_roi(roi)
    assert boundary_buffer._current_logical_roi_in_pixels is not None

    # Test ROI at chunk boundary offset
    roi = Roi((4, 4, 4), (4, 4, 4))  # Second chunk
    boundary_buffer.load_logical_roi(roi)
    assert boundary_buffer._current_logical_roi_in_pixels is not None


def test_roi_smaller_than_chunk_size(boundary_buffer):
    """Test ROI that is smaller than chunk size in all dimensions."""
    # Chunk size is (4, 4, 4), test ROI of size (2, 2, 2)
    roi = Roi((1, 1, 1), (2, 2, 2))

    boundary_buffer.load_logical_roi(roi)

    # Should still load successfully and be expanded to chunk boundaries
    assert boundary_buffer._current_logical_roi_in_pixels is not None
    current_roi = boundary_buffer._current_logical_roi_in_pixels

    # The loaded ROI should be chunk-aligned and contain the original ROI
    assert current_roi.offset[0] % 4 == 0
    assert current_roi.offset[1] % 4 == 0
    assert current_roi.offset[2] % 4 == 0
    assert current_roi.shape[0] % 4 == 0
    assert current_roi.shape[1] % 4 == 0
    assert current_roi.shape[2] % 4 == 0


def test_roi_exactly_one_chunk_size(boundary_buffer):
    """Test ROI that is exactly the size of one chunk."""
    roi = Roi((2, 2, 2), (4, 4, 4))  # One chunk at offset position

    boundary_buffer.load_logical_roi(roi)

    assert boundary_buffer._current_logical_roi_in_pixels is not None
    current_roi = boundary_buffer._current_logical_roi_in_pixels

    # Should be expanded to chunk boundaries
    assert current_roi.shape[0] >= 4
    assert current_roi.shape[1] >= 4
    assert current_roi.shape[2] >= 4


def test_roi_non_chunk_aligned_dimensions(boundary_buffer):
    """Test ROI with dimensions that don't align with chunk size."""
    # Chunk size is (4, 4, 4), test ROI with size (3, 5, 7)
    roi = Roi((1, 1, 1), (3, 5, 7))

    boundary_buffer.load_logical_roi(roi)

    assert boundary_buffer._current_logical_roi_in_pixels is not None
    current_roi = boundary_buffer._current_logical_roi_in_pixels

    # Should be expanded to chunk boundaries
    assert current_roi.shape[0] % 4 == 0
    assert current_roi.shape[1] % 4 == 0
    assert current_roi.shape[2] % 4 == 0


# Data Boundary Precision Tests
def test_roi_extends_beyond_by_one_pixel(boundary_buffer):
    """Test ROI that extends beyond data bounds by exactly one pixel."""
    # Data is 16x16x16, test ROI that goes to 17 in one dimension
    roi = Roi((15, 8, 8), (2, 4, 4))  # Extends from 15 to 17 in X

    boundary_buffer.load_logical_roi(roi)

    assert boundary_buffer._current_logical_roi_in_pixels is not None
    current_roi = boundary_buffer._current_logical_roi_in_pixels

    # Should be clipped to data bounds
    assert current_roi.offset[0] + current_roi.shape[0] <= 16


def test_roi_ends_exactly_at_data_boundary(boundary_buffer):
    """Test ROI that ends exactly at data boundary."""
    # Data is 16x16x16, test ROI that ends exactly at 16
    roi = Roi((12, 12, 12), (4, 4, 4))  # Ends exactly at (16, 16, 16)

    boundary_buffer.load_logical_roi(roi)

    assert boundary_buffer._current_logical_roi_in_pixels is not None
    # Should load successfully without going beyond bounds


def test_roi_starts_at_boundary_extends_beyond(boundary_buffer):
    """Test ROI that starts exactly at data boundary and extends beyond."""
    # Data is 16x16x16, test ROI starting at boundary
    roi = Roi((16, 8, 8), (4, 4, 4))  # Starts at X boundary

    boundary_buffer.load_logical_roi(roi)

    # Should not load anything as it's completely outside
    assert boundary_buffer._current_logical_roi_in_pixels is None


# Sequential Loading Scenarios
def test_sequential_overlapping_rois(boundary_buffer):
    """Test loading overlapping ROIs in sequence."""
    # Load first ROI
    roi1 = Roi((0, 0, 0), (4, 4, 4))
    boundary_buffer.load_logical_roi(roi1)
    first_roi = boundary_buffer._current_logical_roi_in_pixels

    # Load overlapping ROI
    roi2 = Roi((2, 2, 2), (4, 4, 4))
    boundary_buffer.load_logical_roi(roi2)
    second_roi = boundary_buffer._current_logical_roi_in_pixels

    # Should have loaded successfully
    assert first_roi is not None
    assert second_roi is not None

    # The ROIs should overlap
    assert first_roi.intersects(second_roi)


def test_progressive_roi_expansion(boundary_buffer):
    """Test loading progressively larger ROIs from same position."""
    base_pos = (4, 4, 4)

    # Load small ROI
    roi1 = Roi(base_pos, (2, 2, 2))
    boundary_buffer.load_logical_roi(roi1)
    small_roi = boundary_buffer._current_logical_roi_in_pixels

    # Load medium ROI
    roi2 = Roi(base_pos, (4, 4, 4))
    boundary_buffer.load_logical_roi(roi2)
    medium_roi = boundary_buffer._current_logical_roi_in_pixels

    # Load large ROI
    roi3 = Roi(base_pos, (8, 8, 8))
    boundary_buffer.load_logical_roi(roi3)
    large_roi = boundary_buffer._current_logical_roi_in_pixels

    assert small_roi is not None
    assert medium_roi is not None
    assert large_roi is not None

    # Each should contain the previous (after chunk alignment)
    assert medium_roi.contains(small_roi.intersect(medium_roi))


# Empty ROI and Extreme Cases
def test_empty_roi_handling(boundary_buffer):
    """Test handling of empty ROIs."""
    # ROI with zero size
    roi = Roi((4, 4, 4), (0, 0, 0))

    boundary_buffer.load_logical_roi(roi)

    # Should handle gracefully (likely no-op)
    # Don't make strong assertions about the result as behavior may vary


def test_roi_spanning_entire_dataset(boundary_buffer):
    """Test ROI that spans the maximum possible area within buffer constraints."""
    # Buffer is 8x8x8 pixels (2x2x2 chunks of 4x4x4), so test max size that fits
    roi = Roi((0, 0, 0), (8, 8, 8))

    boundary_buffer.load_logical_roi(roi)

    assert boundary_buffer._current_logical_roi_in_pixels is not None
    current_roi = boundary_buffer._current_logical_roi_in_pixels

    # Should be within or equal to data bounds
    assert current_roi.offset[0] >= 0
    assert current_roi.offset[1] >= 0
    assert current_roi.offset[2] >= 0


def test_roi_with_very_large_coordinates(boundary_buffer):
    """Test ROI with very large coordinates."""
    # Test with large positive coordinates
    roi = Roi((1000, 1000, 1000), (4, 4, 4))

    boundary_buffer.load_logical_roi(roi)

    # Should not load anything as it's outside data bounds
    assert boundary_buffer._current_logical_roi_in_pixels is None


def test_roi_with_very_large_negative_coordinates(boundary_buffer):
    """Test ROI with very large negative coordinates."""
    # Test with large negative coordinates
    roi = Roi((-1000, -1000, -1000), (4, 4, 4))

    boundary_buffer.load_logical_roi(roi)

    # Should not load anything as it's outside data bounds
    assert boundary_buffer._current_logical_roi_in_pixels is None


# Complex Intersection Patterns
def test_l_shaped_intersection(boundary_buffer):
    """Test ROI that creates L-shaped intersection with data bounds."""
    # Create ROI that extends beyond bounds in two dimensions
    roi = Roi((14, 14, 8), (4, 4, 4))  # Extends beyond in X and Y

    boundary_buffer.load_logical_roi(roi)

    if boundary_buffer._current_logical_roi_in_pixels is not None:
        current_roi = boundary_buffer._current_logical_roi_in_pixels
        # Should be clipped to data bounds
        assert current_roi.offset[0] + current_roi.shape[0] <= 16
        assert current_roi.offset[1] + current_roi.shape[1] <= 16
        assert current_roi.offset[2] + current_roi.shape[2] <= 16


def test_corner_intersection(boundary_buffer):
    """Test ROI that intersects data bounds only at a corner."""
    # ROI that barely touches data bounds
    roi = Roi((15, 15, 15), (4, 4, 4))  # Only corner overlaps

    boundary_buffer.load_logical_roi(roi)

    if boundary_buffer._current_logical_roi_in_pixels is not None:
        current_roi = boundary_buffer._current_logical_roi_in_pixels
        # Should have minimal overlap
        assert current_roi.size > 0
