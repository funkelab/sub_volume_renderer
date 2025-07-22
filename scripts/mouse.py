import numpy as np
import pygfx as gfx
import tensorstore as ts
import wgpu
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
data = ts.open(
    spec={
        "driver": "zarr",
        "kvstore": "file:///nrs/funke/data/lightsheet/160315_mouse/160315.zarr/raw",
    }
).result()
scaled_data = data[100, 0, 180:565, 300:2010, 350:]
scaled_data = scaled_data.astype(np.float32)
data_chunks = scaled_data.chunk_layout.read_chunk.shape
print(data_chunks)

segmentations = ts.open(
    spec={
        "driver": "zarr",
        "kvstore": "file:///nrs/funke/data/lightsheet/160315_mouse/160315.zarr/segmentation",
    }
).result()
scaled_segmentations = segmentations[100, 0, 180:565, 300:2010, 350:]
scaled_segmentations = scaled_segmentations.astype(np.uint32)
segmentations_chunks = scaled_data.chunk_layout.read_chunk.shape

# create a volume
# noinspection PyTypeChecker
volume = SubVolume(
    SubVolumeMaterial(
        lmip_threshold=150,
        clim=(47, 135),
        lmip_fall_off=0.5,
        lmip_max_samples=25,
        fog_density=0.1,
        fog_color=(0, 0, 0),
        colors=[
            (0.0, 1, 1),
            (0.1, 1, 1),
            (0.2, 1, 1),
            (0.3, 1, 1),
            (0.4, 1, 1),
            (0.5, 1, 1),
            (0.6, 1, 1),
            (0.7, 1, 1),
            (0.8, 1, 1),
            (0.9, 1, 1),
        ],
    ),
    data=scaled_data,
    segmentations=scaled_segmentations,
    buffer_shape_in_chunks=(6, 6, 6),
    chunk_shape_in_pixels=(53, 143, 143),
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
