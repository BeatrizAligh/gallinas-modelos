"""Renders de referencia de los 5 manzanos (frente y lado). blender -b -P vistas_manzanos.py -- carpeta_salida"""
import bpy, os, sys, glob, math
from mathutils import Vector

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(DIR, "_vistas")
os.makedirs(OUT, exist_ok=True)

for path in sorted(glob.glob(os.path.join(DIR, "*.fbx"))):
    name = os.path.splitext(os.path.basename(path))[0].replace(" ", "")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'RANDOM'
    sc.display.shading.show_cavity = True
    sc.render.resolution_x = sc.render.resolution_y = 700
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 1.35
    for i, ang in enumerate([0, 90]):
        a = math.radians(ang)
        cam.location = Vector((math.sin(a) * 4, -math.cos(a) * 4, 0.55))
        cam.rotation_euler = (math.radians(90), 0, a)
        sc.render.filepath = os.path.join(OUT, f"{name}_{i}.png")
        bpy.ops.render.render(write_still=True)
