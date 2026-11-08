"""Inspección de mallas: blender -b -P probe.py -- archivo.fbx ..."""
import bpy, sys
from mathutils import Vector

files = sys.argv[sys.argv.index('--') + 1:]
for f in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=f)
    print('=====', f)
    for o in bpy.data.objects:
        if o.type == 'MESH':
            me = o.data
            ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
            mn = Vector((min(v.x for v in ws), min(v.y for v in ws), min(v.z for v in ws)))
            mx = Vector((max(v.x for v in ws), max(v.y for v in ws), max(v.z for v in ws)))
            mats = [m.name if m else None for m in me.materials]
            print(f"  MESH {o.name}: verts={len(me.vertices)} faces={len(me.polygons)} uv={len(me.uv_layers)} vcol={len(me.color_attributes)} mats={mats}")
            print(f"     min={tuple(round(x,3) for x in mn)} max={tuple(round(x,3) for x in mx)} size={tuple(round(x,3) for x in (mx-mn))} scale={tuple(round(x,3) for x in o.scale)} rot={tuple(round(x,2) for x in o.rotation_euler)}")
        else:
            print(f"  {o.type} {o.name}")
    for m in bpy.data.materials:
        tex = [n.image.name for n in (m.node_tree.nodes if m.use_nodes else []) if n.type == 'TEX_IMAGE' and n.image]
        print('  MAT', m.name, tex)
