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
        color_count="u4",
        # IMPORTANT NOTE
        # wgpu expects each 3xf4 to be padded to 16 bytes, which pygfx will do if we pass in 3xf4 normally
        # however, when we pass in an array of n*3xf4, pygfx does NOT end up padding the vectors!
        # to get around this, we just use an array of n*4xf4 and ignore the last component.
        # all inputs are still expected to be 3-component tuples! this is just a lie we tell pygfx.
        colors="0*4xf4",
    )

    def __init__(
        self,
        lmip_threshold: float,
        lmip_fall_off: float = 0.5,
        lmip_max_samples: int = 10,
        fog_density: float = 0.5,
        fog_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        colors: list[tuple[float, float, float]] | None = None,
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
        if colors is None:
            colors = [
                (0.0, 1.0, 1.0),
                (0.25, 1.0, 1.0),
                (0.5, 1.0, 1.0),
                (0.75, 1.0, 1.0),
            ]
        self.colors = colors

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

    @property
    def _color_count(self) -> float:
        return self.uniform_buffer.data["color_count"]

    @_color_count.setter
    def _color_count(self, value: int) -> None:
        self.uniform_buffer.data["color_count"] = int(value)
        self.uniform_buffer.update_full()

    @property
    def colors(self) -> list[tuple[float, float, float]]:
        # noinspection PyTypeChecker
        return [
            tuple(float(f) for f in plane.flat)
            for plane in self.uniform_buffer.data["colors"]
        ]

    @colors.setter
    def colors(self, colors: list[tuple[float, float, float]]):
        if not isinstance(colors, (tuple, list)):
            raise TypeError("Colors must be a list.")
        colors2 = []
        for color in colors:
            if isinstance(color, (tuple, list)) and len(color) == 3:
                # note that we need to add in an extra alpha component to get a 4xf4 vector!
                # this is because pygfx doesn't pad the 3xf4 vectors in arrays properly!
                # see the note on uniform_type above.
                colors2.append((*color, 1))
            else:
                # Error
                raise TypeError(f"Each color must be an hsv tuple, not {color}")

        # Apply
        self._set_size_of_uniform_array("colors", len(colors2))
        self._color_count = len(colors2)
        for i in range(len(colors2)):
            self.uniform_buffer.data["colors"][i] = colors2[i]
        self.uniform_buffer.update_full()
