"""
Prepara los props para el juego: orientación común (adelante = -Y de Blender, arriba = +Z), tamaño real,
origen en el agarre y exportación FBX con los mismos ejes que la gallina y el gavilán.
blender -b -P preparar_props.py -- <carpeta_salida>
Cada prop lleva su textura <Nombre>-color.png (Retopo/), que se copia junto al FBX como <Nombre>.png.
"""
import bpy, bmesh, sys, os, math, shutil
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index('--') + 1]
os.makedirs(OUT, exist_ok=True)


def load(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=os.path.join(HERE, 'Retopo', name + '.fbx'))
    obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]
    # Aplica la rotación/escala de importación a la malla: queda en coordenadas de mundo (Z arriba).
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    obj.name = obj.data.name = name
    return obj


def verts(obj):
    return np.array([v.co[:] for v in obj.data.vertices])


def bake(obj, m):
    obj.data.transform(m)
    obj.data.update()


# Triangulos objetivo en el juego (los retopos traen ~21 mil: demasiado para algo que va en la mano).
# Decimate por colapso sobre una copia de exportacion; el FBX de Retopo/ no se toca.
OBJETIVO = {'Escopeta': 3500, 'Martillo': 1800, 'Red': 3500}  # el huevo (1,9 mil) se queda: al reducirlo se rompen las costuras UV


def reducir(obj, name):
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    meta = OBJETIVO.get(name)
    if not meta or tris <= meta:
        return tris, tris
    d = obj.modifiers.new('Reducir', 'DECIMATE')
    d.decimate_type = 'COLLAPSE'; d.ratio = meta / tris; d.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=d.name)
    return tris, sum(len(p.vertices) - 2 for p in obj.data.polygons)


def export(obj, name):
    antes, despues = reducir(obj, name)
    print(f'PROP {name}: triangulos {antes} -> {despues}')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, name + '.fbx'), use_selection=True, object_types={'MESH'},
                             apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
                             mesh_smooth_type='FACE', add_leaf_bones=False, bake_anim=False)
    v = verts(obj)
    tex = os.path.join(HERE, 'Retopo', name + '-color.png')
    if os.path.exists(tex):
        shutil.copyfile(tex, os.path.join(OUT, name + '.png'))
    print(f'PROP {name}: verts={len(v)} min={v.min(0).round(3)} max={v.max(0).round(3)}')


def scale_to(obj, axis, length):
    v = verts(obj)
    s = length / (v[:, axis].max() - v[:, axis].min())
    bake(obj, Matrix.Scale(s, 4))


def origin_at(obj, p):
    bake(obj, Matrix.Translation(-Vector(p)))


# ---------------- Escopeta: cañón en +X → hacia delante (-Y); agarre en la empuñadura.
o = load('Escopeta')
bake(o, Matrix.Rotation(math.radians(-90), 4, 'Z'))     # +X → -Y
scale_to(o, 1, 0.78)
v = verts(o)
mn, mx = v.min(0), v.max(0)
origin_at(o, (0.5 * (mn[0] + mx[0]), mx[1] - 0.36 * (mx[1] - mn[1]), mn[2] + 0.42 * (mx[2] - mn[2])))
export(o, 'Escopeta')

# ---------------- Martillo: mango vertical, cabeza arriba; la boca (lado -X) mira adelante (-Y).
o = load('Martillo')
bake(o, Matrix.Rotation(math.radians(90), 4, 'Z'))      # -X → -Y
scale_to(o, 2, 0.42)
v = verts(o)
mn, mx = v.min(0), v.max(0)
origin_at(o, (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), mn[2] + 0.22 * (mx[2] - mn[2])))
export(o, 'Martillo')

# ---------------- Huevo: de pie, 11 cm, origen abajo.
o = load('Huevo')
scale_to(o, 2, 0.11)
v = verts(o)
mn, mx = v.min(0), v.max(0)
origin_at(o, (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), mn[2]))
export(o, 'Huevo')

# ---------------- Red: mango en diagonal → mango hacia arriba (+Z) con el aro arriba; boca del aro hacia delante.
o = load('Red')
v = verts(o)
end = v[np.argmax(v[:, 0] + v[:, 1])]                    # punta del mango (arriba a la derecha en planta)
far = v[np.linalg.norm(v - end, axis=1) > 0.6 * np.linalg.norm(v - end, axis=1).max()]
hoop = far.mean(0)
axis = Vector(hoop - end).normalized()
bake(o, Matrix.Translation(-Vector(end)))
bake(o, axis.rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4())
v = verts(o)
zmid = 0.5 * (v[:, 2].max() + v[:, 2].min())
spread = lambda a: np.ptp(a[:, 0]) + np.ptp(a[:, 1])
if spread(v[v[:, 2] <= zmid]) > spread(v[v[:, 2] > zmid]):   # el aro (lo ancho) tiene que quedar arriba
    bake(o, Matrix.Rotation(math.pi, 4, 'X'))
v = verts(o)
hv = v[v[:, 2] > 0.6 * v[:, 2].max()]
c = hv - hv.mean(0)
w, vec = np.linalg.eigh(c.T @ c)
n = Vector(vec[:, 0]); n.z = 0                         # normal del aro (menor varianza), en horizontal
if n.length > 1e-4:
    ang = math.atan2(n.x, -n.y)
    bake(o, Matrix.Rotation(-ang, 4, 'Z'))              # boca del aro hacia -Y
v = verts(o)
L = v[:, 2].max() - v[:, 2].min()
bake(o, Matrix.Scale(1.0 / L, 4))
v = verts(o)
origin_at(o, (0.0, 0.0, v[:, 2].min() + 0.12))
export(o, 'Red')
