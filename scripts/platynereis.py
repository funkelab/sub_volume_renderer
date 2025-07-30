import numpy as np
import pygfx as gfx
import wgpu
import zarr
from rendercanvas.auto import RenderCanvas, loop
from skimage.measure import regionprops

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
data = zarr.open_array(
    "/Volumes/funke/data/lightsheet/130312_platynereis/13-03-12.zarr/raw"
)
segmentations = zarr.open_array(
    "/Volumes/funke/data/lightsheet/130312_platynereis/13-03-12.zarr/segmentation"
)
# data is stored as [channel, t, z, y, x]
scaled_data = data[0, 378, :, :, :]
scaled_data[:25, :25, :25] = 1.0
scaled_data = scaled_data.astype(np.float32)

# segmentations should match the data slice dimensions
segmentations_data = segmentations[0, 378, :, :, :].astype(np.uint32)

# create a volume
# noinspection PyTypeChecker
volume = SubVolume(
    SubVolumeMaterial(
        lmip_threshold=200,
        lmip_fall_off=0.5,
        lmip_max_samples=25,
        fog_density=0.010,
        fog_color=(0, 0, 0),
        prefer_purple_orange=False,
        clim=(0, np.percentile(scaled_data, 99)),
        gamma=1.0,
        opacity=0.5,
    ),
    data=scaled_data,
    segmentations=segmentations_data,
    buffer_shape_in_chunks=(3, 3, 3),
    # scaled_data is an ndarray now, so we need to provide chunk shape manually
    chunk_shape_in_pixels=data.chunks[2:],
)
volume.world.position = 0, 0, 0
scene.add(volume)

for region in regionprops(segmentations_data):
    geom = gfx.sphere_geometry(0.5)
    material = gfx.MeshBasicMaterial(color=(1, 1, 1, 1))
    mesh = gfx.Mesh(geom, material)
    pixel_loc = np.array([*region.centroid[::-1], 1])
    world_loc = volume.world.matrix @ pixel_loc
    world_loc = tuple(world_loc[:3])
    mesh.world.position = world_loc
    mesh.world.z *= 6

    scene.add(mesh)


volume.world.scale_z = 6


@canvas.request_draw
def do_draw():
    renderer.render(scene, camera)
    # Center the logical ROI on the camera using SubVolume's method
    volume.center_on_position(camera.world.position)


camera.world.position = -19.81, 7.5, 7.5
# noinspection PyTypeChecker
camera.look_at((-1, 0, 0))

loop.run()
