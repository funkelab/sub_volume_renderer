import numpy as np
import pygfx as gfx
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

chunk_0 = np.zeros((16, 16, 16))
chunk_0[:4, :4, :4] = 1
data_0 = np.tile(chunk_0, (16, 16, 16))
chunk_1 = np.zeros((8, 8, 8))
chunk_1[:2, :2, :2] = 1
data_1 = np.tile(chunk_1, (16, 16, 16))
chunk_2 = np.zeros((4, 4, 4))
chunk_2[:1, :1, :1] = 1
data_2 = np.tile(chunk_2, (16, 16, 16))

segmentations_0 = 0 * np.ones((256, 256, 256), dtype=np.uint8)
segmentations_1 = 1 * np.ones((128, 128, 128), dtype=np.uint8)
segmentations_2 = 2 * np.ones((64, 64, 64), dtype=np.uint8)

# create a volume
volume = SubVolume(
    SubVolumeMaterial(
        lmip_threshold=0.5,
        fog_density=0.01,
        colors=[
            (0.0, 1.0, 1.0),
            (0.33, 1.0, 1.0),
            (0.66, 1.0, 1.0),
        ],
    ),
    data_segmentation_pairs=[
        (data_0, segmentations_0),
        (data_1, segmentations_1),
        (data_2, segmentations_2),
    ],
    chunk_shape_in_pixels=[
        (16, 16, 16),
        (8, 8, 8),
        (4, 4, 4),
    ],
    buffer_shape_in_chunks=[
        (2, 2, 2),
        (4, 4, 4),
        (8, 8, 8),
    ],
)
volume.world.position = 0, 0, 0
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
