from ._material import SubVolumeMaterial
from ._wobject import SubVolume
from ._wrapping_buffer import WrappingBuffer

# we need to import the shader module so that the shader is registered
# it's not imported anywhere else
from ._shader import SubVolumeShader  # noqa: F401 # isort: skip

__all__ = [
    "SubVolume",
    "SubVolumeMaterial",
    "WrappingBuffer",
]
