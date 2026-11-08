"""Vídeo de repaso de animaciones de un rig (Workbench, rápido):
blender -b <Nombre>_rig.blend -P video_anim.py -- <textura.png> <carpeta_frames> "Clip:veces,Clip:veces,..."
Deja los fotogramas numerados en <carpeta_frames> (después se codifican con ffmpeg) y un texto con el clip en cada tramo."""
import bpy, sys, os, math
from mathutils import Vector

A = sys.argv[sys.argv.index('--') + 1:]
TEX, OUT, SEL = A[0], A[1], A[2].split(',')
os.makedirs(OUT, exist_ok=True)
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
sc.display.shading.show_shadows = True
R_ = int(os.environ.get('RES', '640'))
sc.render.resolution_x, sc.render.resolution_y = R_, R_
sc.render.film_transparent = False
sc.world = sc.world or bpy.data.worlds.new('W')
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = 1.6
H = max((body.matrix_world @ Vector(c)).z for c in body.bound_box)
target = Vector((0, 0, H * 0.45))
# Suelo para leer el contacto de los pies.
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
floor = bpy.context.active_object
fm = bpy.data.materials.new('F'); fm.diffuse_color = (0.55, 0.72, 0.42, 1); floor.data.materials.append(fm)
labels = open(os.path.join(OUT, 'tramos.txt'), 'w', encoding='utf-8')
k = 0
for item in SEL:
    name, reps = item.split(':')
    act = bpy.data.actions[name]
    arm.animation_data.action = act
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    start = k
    for r in range(int(reps)):
        for f in range(f0, f1 + (0 if r < int(reps) - 1 else 1)):
            # Cámara que gira despacio alrededor (se ve el volumen y los hombros desde varios lados).
            ang = math.radians(-35 + 70 * math.sin(k / 120.0))
            d = Vector((math.sin(ang), -math.cos(ang), 0.28)).normalized()
            cam.location = target + d * 5
            cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
            sc.frame_set(f)
            sc.render.filepath = os.path.join(OUT, f'f{k:05d}.png')
            bpy.ops.render.render(write_still=True)
            k += 1
    labels.write(f'{start} {k - 1} {name}\n')
labels.close()
print('VIDEO frames', k)
