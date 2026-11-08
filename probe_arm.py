import bpy, sys
from mathutils import Vector
f = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=f)
for o in bpy.data.objects:
    if o.type == 'ARMATURE':
        print('ARM', o.name, 'scale', tuple(round(x,3) for x in o.scale), 'bones', len(o.data.bones))
        for b in o.data.bones:
            if b.parent is None or b.name in ('hips','root','talons','head','spine','chest'):
                print('  bone', b.name, 'parent', b.parent.name if b.parent else None, 'head', tuple(round(x,3) for x in (o.matrix_world @ b.head_local)))
    if o.type == 'MESH':
        ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
        print('MESH', o.name, 'size', tuple(round(max(v[i] for v in ws)-min(v[i] for v in ws),3) for i in range(3)), 'minz', round(min(v.z for v in ws),3), 'maxz', round(max(v.z for v in ws),3))
print('ACTIONS', [a.name for a in bpy.data.actions][:30])
