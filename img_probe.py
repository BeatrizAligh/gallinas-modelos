import bpy, sys
f = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=f)
for im in bpy.data.images:
    print('IMG', im.name, 'packed', im.packed_file is not None, 'size', tuple(im.size), 'path', im.filepath)
