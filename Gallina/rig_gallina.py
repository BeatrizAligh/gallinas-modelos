"""Riggea la gallina desde la linea de comandos.

    blender --background --disable-autoexec "gallina prueba modelo_rescate.blend" ^
            --python rig_gallina.py -- "<salida sin extension>"

Genera <salida>.blend y <salida>.fbx (listo para Unity, solo esqueleto + malla, sin animaciones).
Si cambia la malla, ajustar PIEZAS y las coordenadas de huesos: estan medidas en el modelo original.
"""
import bpy, bmesh, sys, os
import numpy as np
from mathutils import Matrix, Vector

out_base = sys.argv[sys.argv.index("--") + 1]

# Referencias medidas en el modelo original (mira a -X; su izquierda es -Y).
CENTRO_Y = 0.0024   # plano de simetria
CENTRO_PIES_X = 0.034

# Centro esperado de cada una de las 16 piezas sueltas de la malla.
PIEZAS = {
    "cuerpo": (-0.050, 0.000, 0.620), "pico": (-0.324, 0.000, 0.789), "cresta": (-0.097, 0.000, 0.941),
    "ojo.L": (-0.216, -0.121, 0.809), "ojo.R": (-0.216, 0.126, 0.809),
    "barbilla.L": (-0.305, -0.034, 0.686), "barbilla.R": (-0.305, 0.037, 0.686),
    "ala.L": (0.051, -0.298, 0.406), "ala.R": (0.050, 0.303, 0.406),
    "pata.L": (0.025, -0.149, 0.028), "pata.R": (0.009, 0.142, 0.029),
    "pluma_centro": (0.496, -0.002, 0.624),
    "pluma_alta.L": (0.534, -0.105, 0.562), "pluma_alta.R": (0.534, 0.100, 0.561),
    "pluma_baja.L": (0.530, -0.136, 0.437), "pluma_baja.R": (0.530, 0.131, 0.437),
}


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def P(x, lado, z):
    """Coordenadas medidas (x, |y|, z) -> espacio del rig. Frente -Y, izquierda +X, pies en el origen."""
    return Vector((lado, x - CENTRO_PIES_X, z))


# ---------------------------------------------------------------- malla
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
to_rig = Matrix.Rotation(np.pi / 2, 4, "Z")
to_rig = Matrix.Translation((CENTRO_Y, -CENTRO_PIES_X, 0)) @ to_rig
original = np.array([list(obj.matrix_world @ v.co) for v in me.vertices])
me.transform(to_rig @ obj.matrix_world)
obj.matrix_world = Matrix.Identity(4)
obj.name, me.name = "Gallina", "Gallina_Mesh"
for mat in me.materials:
    if mat:
        mat.name = "Gallina_Mat"
vco = np.array([list(v.co) for v in me.vertices])   # espacio del rig

# Separa la malla en piezas conectadas e identifica cada una por su centro.
n = len(vco)
adj = [[] for _ in range(n)]
for e in me.edges:
    a, b = e.vertices
    adj[a].append(b); adj[b].append(a)
lab = -np.ones(n, int); n_isl = 0
for i in range(n):
    if lab[i] >= 0:
        continue
    pila = [i]; lab[i] = n_isl
    while pila:
        c = pila.pop()
        for k in adj[c]:
            if lab[k] < 0:
                lab[k] = n_isl; pila.append(k)
    n_isl += 1
assert n_isl == len(PIEZAS), f"esperaba {len(PIEZAS)} piezas, hay {n_isl}"
mask = {}
for isl in range(n_isl):
    c = original[lab == isl].mean(0)
    nombre = min(PIEZAS, key=lambda k: np.linalg.norm(c - np.array(PIEZAS[k])))
    assert nombre not in mask, f"pieza duplicada: {nombre}"
    assert np.linalg.norm(c - np.array(PIEZAS[nombre])) < 0.03, f"pieza lejos de lo esperado: {nombre}"
    mask[nombre] = np.where(lab == isl)[0]

# Ninguna pieza venia cerrada (alas por dentro, cuencas de los ojos, bases ocultas de plumas, patas...).
# Se tapan sin crear vertices, asi los indices y los pesos no cambian. Cada tapa toma la UV de la cara vecina.
bm = bmesh.new(); bm.from_mesh(me)
bm.verts.ensure_lookup_table()
capa_uv = bm.loops.layers.uv.active
abiertos = [e for e in bm.edges if e.is_boundary]
print("HUECOS bordes abiertos antes de tapar", len(abiertos))
nuevas = bmesh.ops.holes_fill(bm, edges=abiertos, sides=0)["faces"]
print("HUECOS tapas creadas", len(nuevas))
vecina = {}
for f in nuevas:
    otra = next(c for e in f.edges for c in e.link_faces if c not in nuevas)
    vecina[f] = otra.loops[0][capa_uv].uv.copy()
for f, uv_vec in vecina.items():
    for l in f.loops:
        l[capa_uv].uv = uv_vec
    f.smooth = True
tris = bmesh.ops.triangulate(bm, faces=nuevas, quad_method="BEAUTY", ngon_method="BEAUTY")["faces"]
print("HUECOS triangulos de tapa", len(tris), "vertices", len(bm.verts), "(sin cambios: %d)" % n)
bm.to_mesh(me); bm.free()

# ---------------------------------------------------------------- esqueleto
# nombre: (padre, cabeza, cola)
HUESOS = {
    "root": (None, Vector((0, 0, 0)), Vector((0, 0, 0.12))),
    "hips": ("root", P(0.18, 0, 0.31), P(0.04, 0, 0.40)),
    "spine": ("hips", P(0.04, 0, 0.40), P(-0.06, 0, 0.50)),
    "neck_01": ("spine", P(-0.06, 0, 0.50), P(-0.125, 0, 0.62)),
    "neck_02": ("neck_01", P(-0.125, 0, 0.62), P(-0.132, 0, 0.69)),
    "head": ("neck_02", P(-0.132, 0, 0.69), P(-0.135, 0, 0.92)),
    "jaw": ("head", P(-0.275, 0, 0.766), P(-0.385, 0, 0.778)),
    "comb": ("head", P(-0.105, 0, 0.875), P(-0.105, 0, 1.03)),
    "tail": ("hips", P(0.30, 0, 0.44), P(0.42, 0, 0.50)),
    "tail_mid": ("tail", P(0.385, 0, 0.56), P(0.593, 0, 0.713)),
}
for s, lado in ((".L", 1.0), (".R", -1.0)):
    HUESOS.update({
        "eye" + s: ("head", P(-0.216, 0.1235 * lado, 0.809), P(-0.216, 0.195 * lado, 0.812)),
        "wing" + s: ("spine", P(-0.16, 0.27 * lado, 0.50), P(0.06, 0.30 * lado, 0.44)),
        "wing_tip" + s: ("wing" + s, P(0.06, 0.30 * lado, 0.44), P(0.275, 0.275 * lado, 0.41)),
        "wattle" + s: ("head", P(-0.296, 0.005 * lado, 0.745), P(-0.314, 0.05 * lado, 0.637)),
        "tail_high" + s: ("tail", P(0.44, 0.107 * lado, 0.44), P(0.61, 0.118 * lado, 0.646)),
        "tail_low" + s: ("tail", P(0.45, 0.10 * lado, 0.35), P(0.588, 0.15 * lado, 0.477)),
        "leg" + s: ("hips", P(0.07, 0.146 * lado, 0.18), P(0.07, 0.146 * lado, 0.055)),
        "foot" + s: ("leg" + s, P(0.07, 0.146 * lado, 0.055), P(-0.015, 0.146 * lado, 0.035)),
        "toes" + s: ("foot" + s, P(-0.015, 0.146 * lado, 0.035), P(-0.11, 0.146 * lado, 0.022)),
    })

arm_data = bpy.data.armatures.new("Gallina_Armature")
arm = bpy.data.objects.new("Gallina_Rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
arm.show_in_front = True
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for nombre, (_, h, t) in HUESOS.items():
    eb = arm_data.edit_bones.new(nombre)
    eb.head, eb.tail = h, t
    eje = (t - h).normalized()
    ref = Vector((0, 0, 1)) if abs(eje.z) < 0.8 else Vector((0, -1, 0))
    eb.align_roll(ref)
for nombre, (padre, h, _) in HUESOS.items():
    if padre:
        eb = arm_data.edit_bones[nombre]
        eb.parent = arm_data.edit_bones[padre]
        eb.use_connect = (eb.parent.tail - h).length < 1e-5
bpy.ops.object.mode_set(mode="OBJECT")

# ---------------------------------------------------------------- pesos
W = {b: np.zeros(n) for b in HUESOS}


def poner(pieza, hueso, peso=1.0):
    W[hueso][mask[pieza]] += peso


def cadena(idx, puntos, huesos, anchos):
    """Mezcla suave a lo largo de una polilinea. Devuelve pesos por hueso para los vertices idx."""
    q = vco[idx]
    pts = [np.array(p) for p in puntos]
    largos = [np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])]
    acum = np.concatenate(([0.0], np.cumsum(largos)))
    mejor_d, s = np.full(len(q), 1e9), np.zeros(len(q))
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        v = b - a
        t = np.clip(((q - a) @ v) / (v @ v), 0, 1)
        d = np.linalg.norm(q - (a + np.outer(t, v)), axis=1)
        m = d < mejor_d
        mejor_d[m], s[m] = d[m], acum[i] + t[m] * largos[i]
    c = [smooth(acum[i + 1] - w, acum[i + 1] + w, s) for i, w in enumerate(anchos)]
    r = {}
    for k, h in enumerate(huesos):
        arriba = c[k - 1] if k > 0 else 1.0
        abajo = c[k] if k < len(c) else 0.0
        r[h] = arriba - abajo
    return r


# Cuerpo: columna continua; la parte trasera sigue a la cola.
idx = mask["cuerpo"]
col = ["hips", "spine", "neck_01", "neck_02", "head"]
r = cadena(idx, [HUESOS[col[0]][1]] + [HUESOS[h][2] for h in col], col, [0.07, 0.06, 0.035, 0.035])
q = vco[idx]
base_cola, dir_cola = np.array(HUESOS["tail"][1]), np.array((0.0, 0.9, 0.43))
peso_cola = smooth(0.0, 0.10, (q - base_cola) @ dir_cola)
huesos_cuerpo = col + ["tail"]
B = np.stack([r[h] * (1 - peso_cola) for h in col] + [peso_cola], axis=1)

# Suaviza sobre la malla: la proyeccion a la columna da saltos entre vertices vecinos en el pecho.
# La cabeza queda fija para que ojos, pico y barbillas no se despeguen de la piel.
local = {int(v): i for i, v in enumerate(idx)}
aristas = np.array([(local[a], local[b]) for a, b in (e.vertices for e in me.edges) if a in local])
fija = r["head"] > 0.999
B_fija = B[fija].copy()
grado = np.zeros(len(idx))
np.add.at(grado, aristas[:, 0], 1); np.add.at(grado, aristas[:, 1], 1)
for _ in range(30):
    suma = np.zeros_like(B)
    np.add.at(suma, aristas[:, 0], B[aristas[:, 1]])
    np.add.at(suma, aristas[:, 1], B[aristas[:, 0]])
    B = 0.5 * B + 0.5 * suma / grado[:, None]
    B[fija] = B_fija
for k, h in enumerate(huesos_cuerpo):
    W[h][idx] += B[:, k]

# Cabeza: ojos rigidos, cresta con base fija y punta libre.
for s in (".L", ".R"):
    poner("ojo" + s, "eye" + s)
    poner("barbilla" + s, "wattle" + s)
idx = mask["cresta"]
pc = smooth(0.87, 0.93, vco[idx, 2])
W["comb"][idx] += pc
W["head"][idx] += 1 - pc

# Pico: lo que queda bajo la linea de la boca va a la mandibula.
idx = mask["pico"]
ori = original[idx]
linea = 0.783 - 0.16 * (ori[:, 0] + 0.3915)
abajo = ori[:, 2] < linea
W["jaw"][idx[abajo]] += 1
W["head"][idx[~abajo]] += 1
print("PICO superior", int((~abajo).sum()), "inferior", int(abajo.sum()))

# El pico es una sola pieza con un pliegue: al abrirlo, las caras que cruzan la boca se estiran.
# Se les apunta la UV a un texel oscuro del ojo para que se vean como el interior de la boca.
img = next((i for i in bpy.data.images if i.size[0] > 256), None)
uv = me.uv_layers.active.data
pix = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
img.pixels.foreach_get(pix)
pix = pix.reshape(img.size[1], img.size[0], 4)
es_ojo = set(mask["ojo.L"].tolist())
mejor_uv, mejor_luz = None, 9.0
for poly in me.polygons:
    if poly.vertices[0] in es_ojo:
        for li in poly.loop_indices:
            u, v = uv[li].uv
            px = pix[int(v * (img.size[1] - 1)) % img.size[1], int(u * (img.size[0] - 1)) % img.size[0]]
            if px[:3].sum() < mejor_luz:
                mejor_luz, mejor_uv = float(px[:3].sum()), (float(u), float(v))
en_pico = {int(v): bool(b) for v, b in zip(idx, abajo)}
cruzan = 0
for poly in me.polygons:
    lados = {en_pico[v] for v in poly.vertices if v in en_pico}
    if len(lados) == 2:
        cruzan += 1
        for li in poly.loop_indices:
            uv[li].uv = mejor_uv
print("PICO caras que cruzan la boca", cruzan, "texel oscuro", mejor_uv, "luz %.3f" % mejor_luz)


# Diagnostico: caras abiertas por pieza (importa porque Unity no dibuja el reverso).
bordes = {}
for poly in me.polygons:
    for ek in poly.edge_keys:
        bordes[ek] = bordes.get(ek, 0) + 1
abiertas = {}
for nombre, ids in mask.items():
    s = set(ids.tolist())
    abiertas[nombre] = sum(1 for (a, b), c in bordes.items() if c == 1 and a in s)
print("BORDES_ABIERTOS", {k: v for k, v in abiertas.items() if v} or "ninguno")

# Cola: cada pluma es rigida con su propio hueso.
poner("pluma_centro", "tail_mid")
for s in (".L", ".R"):
    poner("pluma_alta" + s, "tail_high" + s)
    poner("pluma_baja" + s, "tail_low" + s)
    # Ala
    idx = mask["ala" + s]
    h = ["wing" + s, "wing_tip" + s]
    r = cadena(idx, [HUESOS[h[0]][1], HUESOS[h[0]][2], HUESOS[h[1]][2]], h, [0.07])
    for b in h:
        W[b][idx] += r[b]
    # Pata: tocon -> pie -> dedos delanteros
    idx = mask["pata" + s]
    z, y = vco[idx, 2], vco[idx, 1]
    pierna = smooth(0.055, 0.08, z)
    y_toes = P(-0.015, 0, 0)[1]
    dedos = (1 - pierna) * (1 - smooth(y_toes - 0.015, y_toes + 0.015, y))
    W["leg" + s][idx] += pierna
    W["toes" + s][idx] += dedos
    W["foot" + s][idx] += 1 - pierna - dedos

# Unity usa 4 influencias por vertice por defecto: se conservan las 4 mayores y se renormaliza.
M = np.stack([W[b] for b in HUESOS], axis=1)
orden = np.argsort(M, axis=1)
np.put_along_axis(M, orden[:, :-4], 0.0, axis=1)
M[M < 0.01] = 0.0
M /= M.sum(axis=1, keepdims=True)
W = {b: M[:, i] for i, b in enumerate(HUESOS)}

for nombre, w in W.items():
    idx = np.where(w > 1e-4)[0]
    if len(idx):
        g = obj.vertex_groups.new(name=nombre)
        for i in idx:
            g.add([int(i)], float(w[i]), "REPLACE")

obj.parent = arm
mod = obj.modifiers.new("Armature", "ARMATURE")
mod.object = arm

# ---------------------------------------------------------------- salida
blend, fbx = out_base + ".blend", out_base + ".fbx"
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_mainfile()
bpy.ops.export_scene.fbx(
    filepath=fbx, object_types={"ARMATURE", "MESH"}, use_selection=False,
    apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
    add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
    use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
    bake_anim=False, path_mode="RELATIVE",
)
print("HUESOS", len(HUESOS), "GRUPOS", len(obj.vertex_groups))
print("RIG_OK", blend, fbx)
