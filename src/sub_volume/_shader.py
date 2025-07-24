import pygfx.renderers.wgpu as wgpu
from jinja2 import PackageLoader
from pygfx.renderers.wgpu import (
    BaseShader,
    load_wgsl,
    register_wgsl_loader,
    to_texture_format,
)

# volumeshader internally imports wgpu.ShaderStage to construct the shader stages,
# but wgpu.ShaderStage doesn't exist within pygfx.wgpu for us.
# to get around this, we import the constructed flag directly from volumeshader.
# alternative would be importing from the wgpu module (and not gfx.renderers.wgpu) or
# hard coding the flag.
from pygfx.renderers.wgpu.shaders.volumeshader import vertex_and_fragment

from ._material import SubVolumeMaterial
from ._wobject import SubVolume

register_wgsl_loader("sub_volume", PackageLoader("sub_volume", "shaders"))


@wgpu.register_wgpu_render_function(SubVolume, SubVolumeMaterial)
class SubVolumeShader(wgpu.shaders.volumeshader.VolumeRayShader):
    def __init__(self, wobject: SubVolume, **kwargs):
        # skip the BaseVolumeShader init because we don't have a geometry.grid and
        # that breaks the init.
        # we perform all the actions from BaseVolumeShader inside this init.
        BaseShader.__init__(self, wobject, **kwargs)

        material: SubVolumeMaterial = wobject.material

        # BaseVolumeShader makes a bunch of assertions here about the geometry,
        # but since we require wobject to be a SubVolume,
        # the wobject will have already made those assertions for us.

        # Set the render mode
        # This should always be "mip"
        # Could maybe just remove the parameter from the template
        self["mode"] = "mip"
        # Set image format
        self["climcorrection"] = ""
        fmt = to_texture_format(wobject.textures[0].format)
        if "norm" in fmt or "float" in fmt:
            self["img_format"] = "f32"
            if "unorm" in fmt:
                self["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                self["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            self["img_format"] = "u32"
        else:
            self["img_format"] = "i32"

        # Set gamma
        self["gamma"] = material.gamma

        # Channels
        self["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

        # Colorspace
        self["colorspace"] = wobject.textures[0].colorspace
        if material.map is not None:
            self["colorspace"] = material.map.texture.colorspace

        # Multi-scale support
        self["num_scales"] = len(wobject.wrapping_buffers)

    def get_bindings(self, wobject, shared):
        material = wobject.material

        bindings = [
            wgpu.Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            wgpu.Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            wgpu.Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Bind all scale levels
        for i, buffer in enumerate(wobject.wrapping_buffers):
            # Uniform buffer for this scale
            bindings.append(
                wgpu.Binding(
                    f"u_wrapping_buffer_{i}",
                    "buffer/uniform",
                    buffer.uniform_buffer,
                )
            )

            # Data texture for this scale
            t_scale = wgpu.GfxTextureView(buffer.texture)
            bindings.append(
                wgpu.Binding(
                    f"t_scale_{i}", "texture/auto", t_scale, vertex_and_fragment
                )
            )

            # Segmentations texture for this scale
            t_segmentations_scale = wgpu.GfxTextureView(buffer.segmentations_texture)
            bindings.append(
                wgpu.Binding(
                    f"t_segmentations_scale_{i}",
                    "texture/auto",
                    t_segmentations_scale,
                    vertex_and_fragment,
                )
            )

        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_code(self):
        return load_wgsl(
            "ring_buffer_volume_renderer.wgsl", package_name="sub_volume.shaders"
        )
