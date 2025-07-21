import numpy as np
import pytest
from funlib.geometry import Coordinate

from sub_volume._wrapping_buffer import WrappingBuffer

# For these wrapping buffer tests, we use zero-filled segmentation arrays
# since we are only testing the buffer management functionality and not
# segmentation-specific features.


@pytest.fixture
def chunk_size():
    return Coordinate((4, 4, 4))


@pytest.fixture
def buffer_chunks():
    return Coordinate((5, 5, 5))


@pytest.fixture
def backing_data():
    arr = np.arange(40 * 40 * 40, dtype=np.uint16).reshape((40, 40, 40))
    return arr


@pytest.fixture
def segmentations(backing_data):
    return np.zeros(backing_data.shape, dtype=np.uint16)


@pytest.fixture
def buffer(backing_data, segmentations, chunk_size, buffer_chunks):
    return WrappingBuffer(backing_data, segmentations, buffer_chunks, chunk_size)
