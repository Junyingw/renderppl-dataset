import sys
import os
import pickle
import numpy as np
import bpy
from mathutils import Matrix
import argparse
import math
import random 
import argparse

def reset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)

    # deselect all the objects
    bpy.ops.object.select_all(action="DESELECT")

def scene_setting_init(use_gpu):

    scene = bpy.data.scenes["Scene"]
    scene.render.engine = "CYCLES"
    scene.cycles.use_denoising = True
    scene.cycles.film_transparent = True

    #output
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.use_file_extension = True

    #dimensions
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    
    if use_gpu:
        scene.render.engine = "CYCLES" 
        scene.view_layers[0].cycles.use_denoising = True
        scene.render.tile_x = 512
        scene.render.tile_x = 512
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        for device in cycles_prefs.devices:
            if device.type == "CUDA":
                device.use = True

        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        scene.cycles.device = "GPU" 

               
def node_setting_init():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layer_node = tree.nodes.new("CompositorNodeRLayers")
    img_file_output_node = tree.nodes.new("CompositorNodeOutputFile")
    depth_file_output_node = tree.nodes.new("CompositorNodeOutputFile")
    normal_file_output_node = tree.nodes.new(type="CompositorNodeOutputFile")

    img_file_output_node.format.color_mode = "RGBA"
    img_file_output_node.format.color_depth = "16"
    img_file_output_node.format.file_format = "PNG"
    img_file_output_node.base_path = ""

    depth_file_output_node.format.color_mode = "RGB"
    depth_file_output_node.format.color_depth = "16"
    depth_file_output_node.format.file_format = "OPEN_EXR"
    depth_file_output_node.base_path = ""
    
    normal_file_output_node.format.color_mode = "RGB"
    normal_file_output_node.format.color_depth = "16"
    normal_file_output_node.format.file_format = "OPEN_EXR"
    normal_file_output_node.base_path = ""


    links.new(render_layer_node.outputs["Depth"], depth_file_output_node.inputs[0])
    links.new(render_layer_node.outputs["Image"], img_file_output_node.inputs[0])
    links.new(render_layer_node.outputs["Normal"], normal_file_output_node.inputs[0])
    
def set_lookat(center):
    bpy.ops.mesh.primitive_cube_add(size=1e-10)
    for obj in bpy.data.objects:
        if "Cube" in obj.name:
            cube = obj
           
    # set camera lookat dummy cube 
    cube.location = center
    return cube
    
def set_camera(cam_start, cam_name, lookat):
    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    cam_obj.location=cam_start

    bpy.context.collection.objects.link(cam_obj)

    # add camera to scene
    scene = bpy.context.scene
    scene.camera=cam_obj
    
    constraint = cam_obj.constraints.new(type="TRACK_TO")
    constraint.target=lookat
    
    return cam_obj

def light_setting_init():
    light_data = bpy.data.lights.new(name="mylight", type="SUN")
    light_data.energy = 5.
    light_data.use_shadow = False
    light_object = bpy.data.objects.new(name="my-light", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (0, 0, 1000)

def init_all():
    scene_setting_init(False)
    node_setting_init()
    light_setting_init()
    
def set_save_path(img_path, depth_path, normal_path):
    img_output_node = bpy.context.scene.node_tree.nodes[1]
    depth_output_node = bpy.context.scene.node_tree.nodes[2]
    normal_output_node = bpy.context.scene.node_tree.nodes[3]
    
    img_output_node.base_path = img_path
    depth_output_node.base_path = depth_path
    normal_output_node.base_path = normal_path

def render(view_id, cam_path, cam_obj):
   
    img_file_output_node = bpy.context.scene.node_tree.nodes[1]
    img_file_output_node.file_slots[0].path = "color_###.png" 

    depth_file_output_node = bpy.context.scene.node_tree.nodes[2]
    depth_file_output_node.file_slots[0].path = "depth_###.exr" 
    
    normal_file_output_node = bpy.context.scene.node_tree.nodes[3]
    normal_file_output_node.file_slots[0].path = "normal_###.exr" 
    
    bpy.context.scene.frame_set(view_id + 1)
    bpy.ops.render.render(use_viewport = True, write_still=True)

    # write camera info
    cam_K_file = os.path.join(cam_path, "cam_K.txt")
    K, RT = get_3x4_P_matrix_from_blender(cam_obj)
    np.savetxt(cam_K_file, K)
    np.savetxt(os.path.join(cam_path, "cam_RT_{0:03d}.txt".format(view_id + 1)), RT)

   
def import_model_obj(obj_path, subject_name):
    bpy.ops.import_scene.obj(filepath=obj_path)
    for obj in bpy.data.objects:
        if subject_name in obj.name:
            geo = obj
    return geo

def import_material(geo, base_color_path, material_name=None):

    # Define names and paths
    if material_name is None:
        material_name = "Example_Material"

    # Create a material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    principled_bsdf = nodes.get("Principled BSDF")
    material_output = nodes.get("Material Output")

    # Create Image Texture node and load the base color texture
    if base_color_path is not None:
        base_color = nodes.new("ShaderNodeTexImage")
        base_color.image = bpy.data.images.load(base_color_path)

        # Connect the base color texture to the Principled BSDF
        links.new(principled_bsdf.inputs["Base Color"], base_color.outputs["Color"])
        
    # Assign it to object
    obj = geo
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    current_material = material
    
# convert to blender coords 
def yup_to_zup(x, y, z):
    return x, -z, y

def rotation_matrix(axis, theta):
    new_theta = theta
    rotation_mat = np.array([[math.cos(new_theta), -math.sin(new_theta),0],
                            [math.sin(new_theta), math.cos(new_theta), 0],
                            [0, 0, 1]])
    return rotation_mat


def rotate(point, angle_degrees, axis=(0,1,0)):
    theta_degrees = angle_degrees
    theta_radians = math.radians(theta_degrees)
    rotated_point = np.dot(rotation_matrix(axis, theta_radians), point)
    return rotated_point   

def gen_viewports(center):
    radius = 300.
    half_height = center[-1]
    degree_positive_list = []
    degree_negative_list = []
    for i in range(-61, -1, 12):
        select_degree = random.choice(range(i, i+12))
        degree_negative_list.append(select_degree)
    for i in range(1, 60, 12):
        select_degree = random.choice(range(i, i+12))
        degree_positive_list.append(select_degree)

    degree_list = sorted(degree_negative_list + degree_positive_list + [0])
    cam_positions = []
    
    for idx, degree in enumerate(degree_list):
        height = radius * math.sin(math.radians(degree))
        z = half_height + height
        x = radius * math.cos(math.radians(degree))
        y = 0.
        cam_loc = (x, y, z)
        cam_loc = rotate(cam_loc, idx, axis=(0, 0, 1))
        cam_positions.append(cam_loc)
    return cam_positions

def gen_random_viewport(center):
    radius = 300.
    half_height = center[-1]

    # random sample a degree in range (-60, 60)
    degree = random.uniform(-45, 75)
    height = radius * math.sin(math.radians(degree))
    z = half_height + height
    x = radius * math.cos(math.radians(degree))
    y = 0.
    cam_loc = (x, y, z)

    # random sample a rotation degree in range (0, 360)
    rot = random.uniform(0, 360)
    cam_loc = rotate(cam_loc, rot, axis=(0, 0, 1))
    cam_position = cam_loc
    return cam_position
       
def export_mesh(obj_path, geo):
    bpy.ops.object.select_all(action="DESELECT")
    obj = geo
    obj.select_set(True)
    bpy.ops.export_scene.obj(filepath=obj_path,
                                 check_existing=True,
                                 use_animation=False,
                                 use_normals=True,
                                 use_uvs=True,
                                 use_materials=True, 
                                 use_selection=True)  

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_blender2shapenet = Matrix(
        ((1, 0, 0),
         (0, 0, -1),
         (0, 1, 0)))

    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation

    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam @ R_blender2shapenet
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K, RT


if __name__ == '__main__':

    # change path 
    root_path = "/home/junyingw/renderppl_sample/" 
    out_path = "/home/junyingw/free_view_renderppl" 
    center_file = "./sample_center.txt"

    f = open(center_file, "r")
    center_lines = f.readlines()

    for idx, center_line in enumerate(center_lines):
        center_line = center_lines[idx]
        line = center_line.strip("\n")
        info = line.split(" ")

        # get lookat position (always lookat the center of the human body)
        lookat_center = info[1:]
        subject_name = info[:1][0]

        subject = subject_name + "_OBJ"

        # change to blender coords 
        x, y, z = yup_to_zup(float(lookat_center[0]),float(lookat_center[1]), float(lookat_center[2]))
        lookat_position = (x, y, z)
        cam_org_position = (x+300., y, z)
       
        subject_name = str(subject)[:-4]
        tex_file = os.path.join(root_path, subject, "tex", subject_name + "_dif_2k.jpg")
        obj_file = os.path.join(root_path, subject, subject_name + "_100k.obj")

        os.makedirs(os.path.join(out_path, "image", subject_name),exist_ok=True)
        os.makedirs(os.path.join(out_path, "depth", subject_name),exist_ok=True)
        os.makedirs(os.path.join(out_path, "normal", subject_name),exist_ok=True)
        os.makedirs(os.path.join(out_path, "geometry", subject_name),exist_ok=True)
        os.makedirs(os.path.join(out_path, "camera", subject_name),exist_ok=True)

        img_save_path = os.path.join(out_path, "image", subject_name)
        depth_save_path = os.path.join(out_path, "depth", subject_name)
        normal_save_path = os.path.join(out_path, "normal", subject_name)
        camera_save_path = os.path.join(out_path, "camera", subject_name)
        obj_save_path = os.path.join(out_path, "geometry", subject_name, subject_name + ".obj")

        reset()
        init_all()
        geo = import_model_obj(obj_file, subject_name)
        import_material(geo, tex_file, None)
        export_mesh(obj_save_path, geo)

        lookat = set_lookat(lookat_position)

        view_id = 0

        # if we use random viewports 
        set_save_path(img_save_path, depth_save_path, normal_save_path)

        cam_org_name = "camera_org"
        cam_org = set_camera(cam_org_position, cam_org_name, lookat)
        

        # horizontal rotation 360
        for angle in range(0, 360, 5):
            cam_org_location = cam_org.location
            cam_org.location = rotate(cam_org_location, 5, axis=(0, 0, 1))
            render(view_id, camera_save_path, cam_org)
            view_id += 1

        # random sample viewport    
        for idx, cam_start in enumerate(range(288)):
            cam_name = "camera_{}".format(idx)
            cam_position = gen_random_viewport(lookat_position)
            cam = set_camera(cam_position, cam_name, lookat)
            render(view_id, camera_save_path, cam)
            view_id += 1




