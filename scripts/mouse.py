import numpy as np
import pygfx as gfx
import wgpu
import zarr
from rendercanvas.auto import RenderCanvas, loop

from sub_volume import SubVolume, SubVolumeMaterial

# select gpu
adapters = wgpu.gpu.enumerate_adapters_sync()
description = "Quadro P2000 (DiscreteGPU) via Vulkan"
selected_adapters = [a for a in adapters if description.lower() in a.summary.lower()]
if selected_adapters:
    gfx.renderers.wgpu.select_adapter(selected_adapters[0])

# set up scene
canvas = RenderCanvas(size=(480, 480))
renderer = gfx.renderers.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(fov=45)
controller = gfx.FlyController(camera=camera, register_events=renderer)
scene = gfx.Scene()

# add in background and ambient light
dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
# noinspection PyTypeChecker
background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
scene.add(background)
scene.add(gfx.AmbientLight())

# noinspection SpellCheckingInspection
data = zarr.open_array("/nrs/funke/data/lightsheet/160315_mouse/160315.zarr/raw")
scaled_data = data[100, 0, :, :, :]
scaled_data = scaled_data.astype(np.float32)

# create zero-filled segmentations (not used in this demo)
segmentations = np.zeros(scaled_data.shape, dtype=np.uint16)

# create a volume
# noinspection PyTypeChecker
print(np.max(scaled_data))
volume = SubVolume(
    SubVolumeMaterial(clim=(0, np.max(scaled_data)), map=gfx.cm.inferno),
    data=scaled_data,
    segmentations=segmentations,
    buffer_shape_in_chunks=(3, 3, 3),
    # scaled_data is an ndarray now, so we need to provide chunk shape manually
    chunk_shape_in_pixels=data.chunks[2:],
)
volume.world.position = 0, 0, 0
volume.world.scale_z = 6
scene.add(volume)


@canvas.request_draw
def do_draw():
    renderer.render(scene, camera)
    # Center the logical ROI on the camera using SubVolume's method
    volume.center_on_position(camera.world.position)


camera.world.position = -19.81, 7.5, 7.5
# noinspection PyTypeChecker
camera.look_at((-1, 0, 0))

loop.run()
