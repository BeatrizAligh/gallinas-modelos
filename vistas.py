"""Renders rápidos (Workbench, textura) desde 3 vistas: blender -b -P vistas.py -- modelo.fbx textura.png|- salida_prefijo"""
import bpy, sys, math
from mathutils import Vector

a = sys.argv[sys.argv.index('--') + 1:]
fbx, tex, out = a[0], a[1], a[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
if tex != '-':
    img = bpy.data.images.load(tex)
    mat = bpy.data.materials.new('M'); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes['Principled BSDF']
    ti = nt.nodes.new('ShaderNodeTexImage'); ti.image = img
    nt.links.new(ti.outputs['Color'], bsdf.inputs['Base Color'])
    for o in meshes:
        o.data.materials.clear(); o.data.materials.append(mat)
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr = (mn + mx) / 2; size = max(mx - mn)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'TEXTURE' if tex != '-' else 'MATERIAL'
sc.render.resolution_x = sc.render.resolution_y = 420
sc.render.film_transparent = False
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = size * 1.15
for name, d in [('frente_-Y', Vector((0, -1, 0))), ('lado_+X', Vector((1, 0, 0))), ('arriba_+Z', Vector((0, 0, 1)))]:
    cam.location = ctr + d * size * 3
    cam.rotation_euler = (d * -1).to_track_quat('-Z', 'Y' if name != 'arriba_+Z' else 'Y').to_euler()
    sc.render.filepath = f"{out}_{name}.png"
    bpy.ops.render.render(write_still=True)
print('BOUNDS', tuple(round(x, 3) for x in mn), tuple(round(x, 3) for x in mx))
