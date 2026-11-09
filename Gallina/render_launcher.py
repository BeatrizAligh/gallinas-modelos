"""Renderiza la gallina gigante del launcher (clip Launcher_Intro de animar_gallina.py) con fondo transparente.

    blender -b Gallina_Final_anim.blend -P render_launcher.py -- <carpeta_salida> [tamano=720] [textura]

Deja <carpeta_salida>/f_000.png ... (RGBA, 30 fps) y eventos.json con los momentos que el launcher sincroniza
(pisotones = temblor de pantalla, aterrizaje del giro, pico abierto) y dónde queda el pico en la imagen (de ahí sale
la ventana del launcher). Cámara baja mirando un poco hacia arriba: que parezca enorme.
"""
import bpy, sys, os, json, math
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
OUT = os.path.abspath(args[0])
SIZE = int(args[1]) if len(args) > 1 else 720
TEX = args[2] if len(args) > 2 else os.path.join(os.path.dirname(bpy.data.filepath), "TEXTURAS FINALES", "Gallina_Blanca.png")
os.makedirs(OUT, exist_ok=True)
FPS = 30
CLIP = "Launcher_Intro"
PASOS = (0.3, 0.9, 1.5, 2.1)      # = LANZADOR_PASOS de animar_gallina.py
ATERRIZA, PICO = 2.87, 4.85

sc = bpy.context.scene
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
body = next(o for o in bpy.data.objects if o.type == "MESH")
act = bpy.data.actions[CLIP]
arm.animation_data_create()
for tr in arm.animation_data.nla_tracks:
    tr.mute = True
arm.animation_data.action = act
if hasattr(arm.animation_data, "action_slot") and act.slots:
    arm.animation_data.action_slot = act.slots[0]
f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])

# Material con la textura final (mate, como el resto del juego).
img = bpy.data.images.load(TEX)
mat = bpy.data.materials.new("GallinaLauncher")
mat.use_nodes = True
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
ti = nt.nodes.new("ShaderNodeTexImage")
ti.image = img
nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.62
body.data.materials.clear()
body.data.materials.append(mat)

# Render: Eevee, transparente, cuadrado.
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
sc.render.film_transparent = True
sc.render.resolution_x = sc.render.resolution_y = SIZE
sc.render.resolution_percentage = 100
sc.render.fps = FPS
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.view_settings.view_transform = "Standard"
if sc.world is None:
    sc.world = bpy.data.worlds.new("Mundo")
sc.world.use_nodes = True
bg = next(n for n in sc.world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.95, 0.85, 0.7, 1)
bg.inputs["Strength"].default_value = 0.55

# Medidas de la gallina en reposo.
sc.frame_set(f0)
H = max((body.matrix_world @ Vector(c)).z for c in body.bound_box)

# Luces: sol cálido delante-izquierda, relleno frío y contraluz.
def luz(nombre, tipo, energia, rot, color):
    l = bpy.data.lights.new(nombre, tipo)
    l.energy = energia
    l.color = color
    o = bpy.data.objects.new(nombre, l)
    sc.collection.objects.link(o)
    o.rotation_euler = [math.radians(a) for a in rot]
    return o
luz("Sol", "SUN", 3.2, (55, 0, -35), (1.0, 0.93, 0.82))
luz("Relleno", "SUN", 1.1, (70, 0, 140), (0.75, 0.85, 1.0))
luz("Contra", "SUN", 2.0, (120, 0, 10), (1.0, 0.8, 0.55))

# Cámara baja: el horizonte por debajo de la gallina, mirando un poco hacia arriba.
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 50
dist = H * 2.05
cam.location = Vector((0, -dist, H * 0.12))
objetivo = Vector((0, 0, H * 0.5))
cam.rotation_euler = (objetivo - cam.location).to_track_quat("-Z", "Y").to_euler()

# Render de todos los fotogramas.
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    sc.render.filepath = os.path.join(OUT, f"f_{f - f0:03d}.png")
    bpy.ops.render.render(write_still=True)

# Dónde queda el pico (punta del pico de abajo / cabeza) cuando está más abierto, en coordenadas de la imagen.
from bpy_extras.object_utils import world_to_camera_view
sc.frame_set(f0 + int(round(PICO * FPS)))
dg = bpy.context.evaluated_depsgraph_get()
pb = arm.pose.bones.get("jaw") or arm.pose.bones.get("head")
punta = arm.matrix_world @ pb.tail
v = world_to_camera_view(sc, cam, punta)
json.dump({
    "fps": FPS, "frames": f1 - f0 + 1, "size": SIZE,
    "pasos": list(PASOS), "aterriza": ATERRIZA, "pico": PICO,
    "pico_xy": [round(v.x, 4), round(1 - v.y, 4)],     # 0..1, origen arriba a la izquierda
    "pies_y": round(1 - world_to_camera_view(sc, cam, Vector((0, 0, 0))).y, 4),
}, open(os.path.join(OUT, "eventos.json"), "w"), indent=1)
print("RENDER_OK", f1 - f0 + 1, OUT)
