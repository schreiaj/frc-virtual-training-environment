
import blenderproc as bproc
import argparse
import json
parser = argparse.ArgumentParser()
# parser.add_argument('camera', nargs='?', default="examples/resources/camera_positions", help="Path to the camera file")
parser.add_argument('choreo_path', nargs='?', default="path.traj")
parser.add_argument('scene', nargs='?', default="examples/basics/semantic_segmentation/scene.blend", help="Path to the scene.obj file")
parser.add_argument('output_dir', nargs='?', default="examples/basics/semantic_segmentation/output", help="Path to where the final files, will be saved")
parser.add_argument('--coco', action='store_true')
parser.add_argument('--gif', action='store_true')
args = parser.parse_args()

bproc.init()

with open(args.choreo_path, "r") as f:
    traj = json.load(f)
    for sample in traj['samples']:
        position = [sample['x'], sample['y'], 0.3]
        euler_rotation = [1.5708, 0, sample['heading'] - 1.5708]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)



# load the objects into the scene
objs = bproc.loader.load_blend(args.scene)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([2, 4, 5])
light.set_energy(3000)

light2 = bproc.types.Light()
light2.set_type("POINT")
light2.set_location([8, 4, 5])
light2.set_energy(3000)


light3 = bproc.types.Light()
light3.set_type("POINT")
light3.set_location([14, 4, 5])
light3.set_energy(3000)


# print(len(bpy.data.collections))


# notes = bpy.data.collections['Note'].objects

notes = bproc.filter.by_attr(objs, "name", "Object.*", regex=True) 

for j, obj in enumerate(objs):
    obj.set_cp("category_id", 0)

# Set some category ids for loaded objects
for j, obj in enumerate(notes):
    obj.set_cp("category_id", 1)
    

#  define the camera intrinsics
bproc.camera.set_resolution(640, 640)

# read the camera positions file and convert into homogeneous camera-world transformation
# with open(args.camera, "r") as f:
#     for line in f.readlines():
#         line = [float(x) for x in line.split()]
#         position, euler_rotation = line[:3], line[3:6]
#         matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
#         bproc.camera.add_camera_pose(matrix_world)



# activate depth rendering
# bproc.renderer.enable_depth_output(activate_antialiasing=False)
# enable segmentation masks (per class and per instance)
bproc.renderer.enable_segmentation_output(
    map_by=["instance", "class", "name"], 
    default_values={'category_id': 0}, 
    pass_alpha_threshold=0.5)

# render the whole pipeline
data = bproc.renderer.render()

import os
# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)

if args.gif:
    bproc.writer.write_gif_animation(args.output_dir, data, frame_duration_in_ms=1/60.0)

# write to a coco file format
if args.coco:
    bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")