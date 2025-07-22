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
        fmt = to_texture_format(wobject.texture.format)
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
        self["colorspace"] = wobject.texture.colorspace
        if material.map is not None:
            self["colorspace"] = material.map.texture.colorspace

    def get_bindings(self, wobject, shared):
        material = wobject.material

        bindings = [
            wgpu.Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            wgpu.Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            wgpu.Binding("u_material", "buffer/uniform", material.uniform_buffer),
            wgpu.Binding(
                "u_wrapping_buffer",
                "buffer/uniform",
                wobject.wrapping_buffer.uniform_buffer,
            ),
        ]

        # our ring buffer. iffy on how the sampler should work, but in theory the only place where interpolation matters
        # would be in the boundary where the ring buffer in a single dimension ends. still, we only use textureLoad and
        # don't use the sampler in the shader yet.
        t_ring_buffer = wgpu.GfxTextureView(wobject.texture)
        s_ring_buffer = wgpu.GfxSampler(
            # material.interpolation,
            "nearest",
            "repeat",
        )
        bindings.append(
            wgpu.Binding(
                "s_ring_buffer", "sampler/filtering", s_ring_buffer, "FRAGMENT"
            )
        )
        bindings.append(
            wgpu.Binding(
                "t_ring_buffer", "texture/auto", t_ring_buffer, vertex_and_fragment
            )
        )

        # segmentations ring buffer
        t_segmentations_ring_buffer = wgpu.GfxTextureView(wobject.segmentations_texture)
        bindings.append(
            wgpu.Binding(
                "t_segmentations_ring_buffer",
                "texture/auto",
                t_segmentations_ring_buffer,
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
