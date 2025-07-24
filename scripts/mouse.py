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
scaled_data_0 = data[100, 0, 180:565, 300:2010, 350:]
scaled_data_0 = scaled_data_0.astype(np.float32)
scaled_data_chunks_0 = scaled_data_0.chunk_layout.read_chunk.shape

scaled_data_1 = scaled_data_0[::2, ::2, ::2]
scaled_data_chunks_1 = scaled_data_1.chunk_layout.read_chunk.shape

scaled_data_2 = scaled_data_1[::2, ::2, ::2]
scaled_data_chunks_2 = scaled_data_2.chunk_layout.read_chunk.shape

segmentations = ts.open(
    spec={
        "driver": "zarr",
        "kvstore": "file:///nrs/funke/data/lightsheet/160315_mouse/160315.zarr/segmentation",
    }
).result()
scaled_segmentations_0 = segmentations[100, 0, 180:565, 300:2010, 350:]
scaled_segmentations_0 = scaled_segmentations_0.astype(np.uint32)
scaled_segmentations_chunks_0 = (53, 143, 143)

scaled_segmentations_1 = scaled_segmentations_0[::2, ::2, ::2]
scaled_segmentations_chunks_1 = scaled_data_chunks_1

scaled_segmentations_2 = scaled_segmentations_1[::2, ::2, ::2]
scaled_segmentations_chunks_2 = scaled_data_chunks_2

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
            (0.0, 1.0, 1.0),
            (0.33, 1.0, 1.0),
            (0.66, 1.0, 1.0),
        ],
    ),
    data_segmentation_pairs=[
        (scaled_data_0, scaled_segmentations_0),
        (scaled_data_1, scaled_segmentations_1),
        (scaled_data_2, scaled_segmentations_2),
    ],
    chunk_shape_in_pixels=[
        scaled_data_chunks_0,
        scaled_data_chunks_1,
        scaled_data_chunks_2,
    ],
    buffer_shape_in_chunks=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
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
