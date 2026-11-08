"""Vídeo de los 5 manzanos con su Idle (Workbench). blender -b -P video_manzanos.py -- <carpeta_frames>
Primero los 5 juntos (2 vueltas del bucle) y después un primer plano de cada uno (1 vuelta).
Color de repaso: verde = lo que mueven los huesos de las hojas, marrón = quieto (tronco, hierba, base)."""
import bpy, os, sys, glob, math
from mathutils import Vector

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)
LOOP = 120

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
names = sorted(os.path.basename(p)[:-11] for p in glob.glob(os.path.join(DIR, "*_anim.blend")))
trees = []
for i, n in enumerate(names):
    with bpy.data.libraries.load(os.path.join(DIR, n + "_anim.blend")) as (src, dst):
        dst.objects = [o for o in src.objects]
    arm = mesh = None
    for o in dst.objects:
        sc.collection.objects.link(o)
        if o.type == 'ARMATURE':
            arm = o
        elif o.type == 'MESH':
            mesh = o
    arm.location.x = (i - 2) * 1.6
    # Color por peso: verde lo que se mueve, marrón lo quieto.
    raiz = mesh.vertex_groups["Raiz"].index
    col = mesh.data.color_attributes.new("repaso", 'FLOAT_COLOR', 'POINT')
    for v in mesh.data.vertices:
        w = next((g.weight for g in v.groups if g.group == raiz), 0.0)
        col.data[v.index].color = (0.36 + 0.24 * w, 0.62 - 0.32 * w, 0.24 + 0.02 * w, 1)
    trees.append((n, arm, mesh))

sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'VERTEX'
sc.display.shading.show_shadows = False
sc.world = bpy.data.worlds.new('W')
sc.world.color = (0.62, 0.78, 0.95)
sc.display.shading.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.fps = 30
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
floor = bpy.context.active_object
floor.data.color_attributes.new("repaso", 'FLOAT_COLOR', 'POINT')
for c in floor.data.color_attributes[0].data:
    c.color = (0.62, 0.74, 0.46, 1)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 50


def aim(target, dist, elev, yaw):
    d = Vector((math.sin(yaw) * math.cos(elev), -math.cos(yaw) * math.cos(elev), math.sin(elev)))
    cam.location = target + d * dist
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()


tramos = open(os.path.join(OUT, "tramos.txt"), "w", encoding="utf-8")
k = 0
shots = [("Los 5", Vector((0, 0, 0.42)), 8.6, None, 2 * LOOP)]
for i, (n, arm, mesh) in enumerate(trees):
    shots.append((n, Vector((arm.location.x, 0, 0.55)), 2.9, i, LOOP))
for label, target, dist, idx, frames in shots:
    start = k
    for f in range(frames):
        sc.frame_set(f % LOOP + 1)
        yaw = math.radians(-8 + 16 * f / frames) if idx is None else math.radians(-30 + 60 * f / frames)
        aim(target, dist, math.radians(12), yaw)
        sc.render.filepath = os.path.join(OUT, f"f{k:05d}.png")
        if not os.path.exists(sc.render.filepath):
            bpy.ops.render.render(write_still=True)
        k += 1
    tramos.write(f"{start} {k - 1} {label}\n")
tramos.close()
print("VIDEO frames", k)
