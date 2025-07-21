import numpy as np
import pygfx as gfx


class SubVolumeMaterial(gfx.VolumeMipMaterial):
    uniform_type = dict(
        gfx.VolumeMipMaterial.uniform_type,
        lmip_threshold="f4",
        lmip_fall_off="f4",
        # this should be a u4, but the existing shader math already assumes i4
        lmip_max_samples="i4",
        fog_density="f4",
        fog_color="3xf4",
    )

    def __init__(
        self,
        lmip_threshold: float,
        lmip_fall_off: float = 0.5,
        lmip_max_samples: int = 10,
        fog_density: float = 0.5,
        fog_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        prefer_purple_orange: bool = False,
        clim: tuple[float, float] = (0, 1),
        gamma: float = 1.0,
        opacity: float = 1.0,
    ):
        # we need to super init so gfx.Material will create our uniform buffer
        super().__init__(
            clim=clim,
            gamma=gamma,
            opacity=opacity,
            # we want to force using depth
            depth_test=True,
        )
        self.lmip_threshold = lmip_threshold
        self.lmip_fall_off = lmip_fall_off
        self.lmip_max_samples = lmip_max_samples
        self.fog_density = fog_density
        self.fog_color = fog_color
        self.prefer_purple_orange = prefer_purple_orange

    @property
    def lmip_threshold(self) -> float:
        """The minimum intensity considered significant for the LMIP algorithm."""
        return self.uniform_buffer.data["lmip_threshold"]

    @lmip_threshold.setter
    def lmip_threshold(self, value: float) -> None:
        self.uniform_buffer.data["lmip_threshold"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def lmip_fall_off(self) -> float:
        """The fraction of the maximum intensity that is still considered significant for the LMIP algorithm."""
        return self.uniform_buffer.data["lmip_fall_off"]

    @lmip_fall_off.setter
    def lmip_fall_off(self, value: float) -> None:
        self.uniform_buffer.data["lmip_fall_off"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def lmip_max_samples(self) -> int:
        """The maximum number of samples to consider after detecting a significant intensity."""
        return self.uniform_buffer.data["lmip_max_samples"]

    @lmip_max_samples.setter
    def lmip_max_samples(self, value: int) -> None:
        self.uniform_buffer.data["lmip_max_samples"] = int(value)
        self.uniform_buffer.update_full()

    @property
    def fog_density(self) -> float:
        """The density of the fog effect applied to the volume."""
        return self.uniform_buffer.data["fog_density"]

    @fog_density.setter
    def fog_density(self, value: float) -> None:
        self.uniform_buffer.data["fog_density"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def fog_color(self) -> tuple[float, float, float]:
        """The color of the fog effect applied to the volume."""
        # noinspection PyTypeChecker
        return tuple(self.uniform_buffer.data["fog_color"])

    @fog_color.setter
    def fog_color(self, fog_color: tuple[float, float, float]) -> None:
        """Set the color of the fog effect applied to the volume."""
        if len(fog_color) != 3:
            raise ValueError("fog_color must be a tuple of three floats (r, g, b)")
        if not all(isinstance(c, (int, float)) for c in fog_color):
            raise ValueError("fog_color must contain only numeric values")
        fog_color = np.array(fog_color, dtype=np.float32)
        if np.any(fog_color < 0) or np.any(fog_color > 1):
            raise ValueError("fog_color values must be in the range [0, 1]")
        self.uniform_buffer.data["fog_color"] = fog_color
        self.uniform_buffer.update_full()
