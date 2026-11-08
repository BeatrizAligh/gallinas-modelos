"""Hoja de poses de un rig: blender -b <Nombre>_rig.blend -P hoja_poses.py -- <textura.png> <salida.png> [clip:t,clip:t,...]"""
import bpy, sys, math, os
from mathutils import Vector

A = sys.argv[sys.argv.index('--') + 1:]
TEX, OUT = A[0], A[1]
SEL = A[2].split(',') if len(A) > 2 else None
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
body = [o for o in bpy.data.objects if o.type == 'MESH'][0]
img = bpy.data.images.load(TEX)
mat = bpy.data.materials.new('M'); mat.use_nodes = True
ti = mat.node_tree.nodes.new('ShaderNodeTexImage'); ti.image = img
bsdf = [n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
mat.node_tree.links.new(ti.outputs['Color'], bsdf.inputs['Base Color'])
body.data.materials.clear(); body.data.materials.append(mat)
for tr in arm.animation_data.nla_tracks: tr.mute = True
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'TEXTURE'
sc.render.resolution_x = sc.render.resolution_y = 300
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = float(os.environ.get('ORTHO', '1.9'))
H = max((body.matrix_world @ Vector(c)).z for c in body.bound_box)
target = Vector((0, 0, H * 0.45))
d = Vector((0.75, -1, 0.35)).normalized()
cam.location = target + d * 5
cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
if SEL is None:
    SEL = []
    for act in bpy.data.actions:
        SEL += [f'{act.name}:0.0', f'{act.name}:0.5']
tiles = []
for i, item in enumerate(SEL):
    name, t = item.split(':')
    act = bpy.data.actions[name]
    arm.animation_data.action = act
    f0, f1 = act.frame_range
    sc.frame_set(int(round(f0 + (f1 - f0) * float(t))))
    path = os.path.join(os.path.dirname(OUT), f'_t{i:03d}.png')
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    tiles.append((path, item))
# Hoja: imágenes en cuadrícula con su nombre.
cols = 8
rows = (len(tiles) + cols - 1) // cols
sheet = bpy.data.images.new('sheet', 300 * cols, 300 * rows)
import numpy as np
buf = np.zeros((300 * rows, 300 * cols, 4), dtype=np.float32)
for i, (p, item) in enumerate(tiles):
    im = bpy.data.images.load(p)
    a = np.array(im.pixels[:], dtype=np.float32).reshape(300, 300, 4)
    r, c = rows - 1 - i // cols, i % cols
    buf[r * 300:(r + 1) * 300, c * 300:(c + 1) * 300] = a
    os.remove(p)
sheet.pixels = buf.ravel()
sheet.filepath_raw = OUT; sheet.file_format = 'PNG'; sheet.save()
print('HOJA', OUT, [t[1] for t in tiles])
