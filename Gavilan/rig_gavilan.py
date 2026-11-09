"""Riggea el gavilan desde la linea de comandos.

    blender --background --factory-startup --python rig_gavilan.py -- "<salida sin extension>" "Final/Gavilan_Final.fbx" "Final/Gavilan_Cafe.png"

Genera <salida>.blend y <salida>.fbx (malla + esqueleto, sin animaciones). No modifica el FBX ni las texturas.
Despues: animar_gavilan.py sobre <salida>.blend. El rig intermedio no se guarda en el repo: solo queda Gavilan_Final_anim.
El espacio del rig es el del modelo: mira a -Y, su izquierda es +X y los pies estan en z = 0.
Modelo final (2026-11-08): la mandibula es una pieza aparte y ya no hay huecos en los hombros.
Si cambia la malla hay que volver a medir las coordenadas de HUESOS (las de dedos, pico, ojos y cola salen de la malla).
"""
import bpy, bmesh, sys, os
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
out_base = args[0]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=os.path.abspath(args[1]))
img = bpy.data.images.load(os.path.abspath(args[2]), check_existing=True)
for m in bpy.data.materials:
    for nd in (m.node_tree.nodes if m.node_tree else []):
        if nd.type == "TEX_IMAGE":     # la interfaz de Blender esta en espanol: los nodos se buscan por tipo
            nd.image = img
for i in list(bpy.data.images):
    if i != img and i.users == 0:
        bpy.data.images.remove(i)


def smooth(a, b, x):
    """0 en a, 1 en b. Tambien sirve con a > b para una bajada."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def V(x, y, z):
    return Vector((float(x), float(y), float(z)))


# ---------------------------------------------------------------- malla
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
me.transform(obj.matrix_world)
_xs = [v.co.x for v in me.vertices]
CENTRO_X = (min(_xs) + max(_xs)) / 2   # plano de simetria
me.transform(Matrix.Translation((-CENTRO_X, 0, 0)))
obj.matrix_world = Matrix.Identity(4)
obj.name, me.name = "Gavilan", "Gavilan_Mesh"
for mat in me.materials:
    if mat:
        mat.name = "Gavilan_Mat"
print("MALLA centro x", round(CENTRO_X, 5))

pix = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
img.pixels.foreach_get(pix)
pix = pix.reshape(img.size[1], img.size[0], 4)


def texel(uv):
    return pix[int(uv[1] * (img.size[1] - 1)) % img.size[1], int(uv[0] * (img.size[0] - 1)) % img.size[0]][:3]


def islas(me):
    n = len(me.vertices)
    adj = [[] for _ in range(n)]
    for e in me.edges:
        a, b = e.vertices
        adj[a].append(b); adj[b].append(a)
    lab = -np.ones(n, int); k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        pila = [i]; lab[i] = k
        while pila:
            c = pila.pop()
            for j in adj[c]:
                if lab[j] < 0:
                    lab[j] = k; pila.append(j)
        k += 1
    return lab, k, adj


# Huecos: solo se tapan los del cuerpo y las alas. Los ojos y las motas son conchas abiertas por detras, pegadas a la piel.
lab, n_isl, adj = islas(me)
grandes = set(int(i) for i in np.argsort(-np.bincount(lab))[:3])
bm = bmesh.new(); bm.from_mesh(me)
bm.verts.ensure_lookup_table()
capa_uv = bm.loops.layers.uv.active
abiertos = [e for e in bm.edges if e.is_boundary and lab[e.verts[0].index] in grandes]
print("HUECOS bordes abiertos en cuerpo y alas", len(abiertos))
nuevas = bmesh.ops.holes_fill(bm, edges=abiertos, sides=0)["faces"] if abiertos else []
es_nueva = set(nuevas)
# El atlas tiene fondo negro: cada tapa usa el texel mediano de sus vecinas, ni el negro de la junta ni un claro suelto.
for f in nuevas:
    uvs = [l[capa_uv].uv.copy() for e in f.edges for c in e.link_faces if c not in es_nueva for l in c.loops]
    uvs.sort(key=lambda u: float(texel(u).sum()))
    for l in f.loops:
        l[capa_uv].uv = uvs[len(uvs) // 2]
    f.smooth = True
if nuevas:
    bmesh.ops.triangulate(bm, faces=nuevas, quad_method="BEAUTY", ngon_method="BEAUTY")
print("HUECOS tapas", len(nuevas))
bm.to_mesh(me); bm.free()
n = len(me.vertices)
vco = np.array([list(v.co) for v in me.vertices])

lab, n_isl, adj = islas(me)
print("MALLA piezas", n_isl)
tam = np.bincount(lab)
orden_isl = np.argsort(-tam)
i_cuerpo = orden_isl[0]
i_a, i_b = orden_isl[1], orden_isl[2]
i_ala = {".L": i_a if vco[lab == i_a, 0].mean() > 0 else i_b, ".R": i_b if vco[lab == i_a, 0].mean() > 0 else i_a}
cuerpo = np.where(lab == i_cuerpo)[0]
ala_idx = {s: np.where(lab == i_ala[s])[0] for s in i_ala}
otras = [i for i in range(n_isl) if i not in (i_cuerpo, i_a, i_b)]
ojo, pico_bajo = {}, None
for isl in otras:
    c = vco[lab == isl].mean(0)
    if c[2] > 0.5 and abs(c[0]) < 0.01 and c[1] < -0.09:
        pico_bajo = np.where(lab == isl)[0]          # la mandibula: pieza central bajo el pico
    elif c[2] > 0.55 and abs(c[0]) > 0.02:
        ojo[".L" if c[0] > 0 else ".R"] = np.where(lab == isl)[0]
assert len(ojo) == 2 and pico_bajo is not None, "no encuentro los ojos o la mandibula"
centro_ojo = {s: vco[ojo[s]].mean(0) for s in ojo}
adornos = [isl for isl in otras if vco[lab == isl].mean(0)[2] <= 0.5]   # motas del pecho y de las alas
print("MALLA motas", len(adornos))

# Mandibula: bisagra en su extremo trasero, punta en el delantero.
pj = vco[pico_bajo]
atras = pj[pj[:, 1] > pj[:, 1].max() - 0.012]
delante = pj[pj[:, 1] < pj[:, 1].min() + 0.012]
JAW_H = V(0, pj[:, 1].max() - 0.004, atras[:, 2].mean())
JAW_T = V(0, pj[:, 1].min(), delante[:, 2].mean())
print("PICO mandibula", tuple(round(x, 3) for x in JAW_H), "->", tuple(round(x, 3) for x in JAW_T), len(pico_bajo), "vertices")

# Cola: base fija sobre la rabadilla; la punta es el vertice mas lejano hacia atras.
BASE_COLA = np.array((0.0, 0.13, 0.265))
PUNTA_COLA = vco[cuerpo][np.argmax(vco[cuerpo][:, 1])].copy(); PUNTA_COLA[0] = 0.0
LARGO_COLA = float(np.linalg.norm(PUNTA_COLA - BASE_COLA))
DIR_COLA = (PUNTA_COLA - BASE_COLA) / LARGO_COLA
MEDIO_COLA = BASE_COLA + DIR_COLA * LARGO_COLA * 0.5
print("COLA punta", PUNTA_COLA.round(3), "largo", round(LARGO_COLA, 3))

# ---------------------------------------------------------------- esqueleto
# Dedos: 1 = trasero, 2 = interior, 3 = central, 4 = exterior. (x, y) de la punta en la pata izquierda.
CENTRO_PIE = np.array([0.088, -0.005])
DEDOS = {"1": (0.051, 0.101), "2": (0.046, -0.141), "3": (0.154, -0.141), "4": (0.214, -0.068)}
# La punta exacta de cada dedo se toma de la malla (el vertice mas lejano de su sector).
mp = vco[cuerpo]
pie_iz = mp[(mp[:, 0] > 0) & (mp[:, 2] < 0.075) & (mp[:, 1] < 0.15)]
d_pie = pie_iz[:, :2] - CENTRO_PIE
for k, (lo, hi) in {"1": (60, 140), "2": (-140, -92), "3": (-88, -48), "4": (-45, 30)}.items():
    ang = np.degrees(np.arctan2(d_pie[:, 1], d_pie[:, 0]))
    sel = (ang >= lo) & (ang < hi)
    r = np.linalg.norm(d_pie[sel], axis=1)
    DEDOS[k] = tuple(float(x) for x in pie_iz[sel][np.argmax(r)][:2])
print("DEDOS", {k: tuple(round(x, 3) for x in v) for k, v in DEDOS.items()})

HUESOS = {
    "root": (None, V(0, 0, 0), V(0, 0, 0.12)),
    "hips": ("root", V(0, -0.01, 0.30), V(0, -0.01, 0.36)),
    "spine": ("hips", V(0, -0.01, 0.36), V(0, -0.012, 0.42)),
    "chest": ("spine", V(0, -0.012, 0.42), V(0, -0.015, 0.47)),
    "neck_01": ("chest", V(0, -0.015, 0.47), V(0, -0.018, 0.51)),
    "neck_02": ("neck_01", V(0, -0.018, 0.51), V(0, -0.02, 0.55)),
    "head": ("neck_02", V(0, -0.02, 0.55), V(0, -0.02, 0.70)),
    "jaw": ("head", JAW_H, JAW_T),
    "crest": ("head", V(0, 0.02, 0.60), V(0, 0.14, 0.64)),
    "tail_01": ("hips", V(*BASE_COLA), V(*MEDIO_COLA)),
    "tail_02": ("tail_01", V(*MEDIO_COLA), V(*PUNTA_COLA)),
    # Hueso sin peso: en Unity es el punto donde se sujeta la presa.
    "talons": ("hips", V(0, -0.005, 0.03), V(0, -0.005, -0.05)),
}
for s, k in ((".L", 1.0), (".R", -1.0)):
    HUESOS.update({
        "eye" + s: ("head", V(*centro_ojo[s]), V(*centro_ojo[s]) + V(0.035 * k, -0.026, 0)),
        "brow" + s: ("head", V(0.04 * k, -0.105, 0.645), V(0.105 * k, -0.075, 0.635)),
        "wing_01" + s: ("chest", V(0.15 * k, -0.03, 0.41), V(0.29 * k, -0.055, 0.43)),
        "wing_02" + s: ("wing_01" + s, V(0.29 * k, -0.055, 0.43), V(0.40 * k, -0.072, 0.43)),
        "wing_03" + s: ("wing_02" + s, V(0.40 * k, -0.072, 0.43), V(0.47 * k, -0.084, 0.425)),
        "wing_04" + s: ("wing_03" + s, V(0.47 * k, -0.084, 0.425), V(0.553 * k, -0.098, 0.415)),
        "feather_1" + s: ("wing_01" + s, V(0.22 * k, -0.04, 0.41), V(0.22 * k, 0.012, 0.25)),
        "feather_2" + s: ("wing_02" + s, V(0.345 * k, -0.06, 0.41), V(0.345 * k, -0.025, 0.25)),
        "feather_3" + s: ("wing_03" + s, V(0.435 * k, -0.075, 0.41), V(0.45 * k, -0.06, 0.29)),
        "tail_fan" + s: ("tail_01", V(0.03 * k, 0.17, 0.225), V(0.10 * k, PUNTA_COLA[1] - 0.01, PUNTA_COLA[2] + 0.015)),
        "leg" + s: ("hips", V(0.078 * k, -0.005, 0.19), V(0.089 * k, -0.006, 0.098)),
        "foot" + s: ("leg" + s, V(0.089 * k, -0.006, 0.098), V(0.088 * k, -0.005, 0.04)),
    })
    for d, (px, py) in DEDOS.items():
        c = Vector((CENTRO_PIE[0], CENTRO_PIE[1]))
        p = Vector((px, py))
        u = (p - c).normalized()
        a, b = c + u * 0.03, c + u * 0.088
        HUESOS["toe_%s%s" % (d, s)] = ("foot" + s, V(a.x * k, a.y, 0.04), V(b.x * k, b.y, 0.036))
        HUESOS["claw_%s%s" % (d, s)] = ("toe_%s%s" % (d, s), V(b.x * k, b.y, 0.036), V(p.x * k, p.y, 0.015))

arm_data = bpy.data.armatures.new("Gavilan_Armature")
arm = bpy.data.objects.new("Gavilan_Rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
arm.show_in_front = True
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for nombre, (_, h, t) in HUESOS.items():
    eb = arm_data.edit_bones.new(nombre)
    eb.head, eb.tail = h, t
    eje = (t - h).normalized()
    eb.align_roll(Vector((0, 0, 1)) if abs(eje.z) < 0.8 else Vector((0, -1, 0)))
for nombre, (padre, h, _) in HUESOS.items():
    if padre:
        eb = arm_data.edit_bones[nombre]
        eb.parent = arm_data.edit_bones[padre]
        eb.use_connect = (eb.parent.tail - h).length < 1e-5
bpy.ops.object.mode_set(mode="OBJECT")
arm_data.bones["talons"].use_deform = False

# ---------------------------------------------------------------- pesos
W = {b: np.zeros(n) for b in HUESOS}


def cadena(q, puntos, huesos, anchos):
    """Mezcla suave a lo largo de una polilinea. Devuelve (pesos por hueso, distancia a la linea)."""
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
    return r, mejor_d


# Alas: cuatro huesos a lo largo; el borde de fuga pasa a los huesos de cuerda, que pliegan las plumas.
for s in (".L", ".R"):
    idx = ala_idx[s]
    q = vco[idx]
    span = ["wing_01" + s, "wing_02" + s, "wing_03" + s, "wing_04" + s]
    r, _ = cadena(q, [HUESOS[span[0]][1]] + [HUESOS[h][2] for h in span], span, [0.035, 0.03, 0.03])
    ax = np.abs(q[:, 0])
    linea_z = np.interp(ax, [0.15, 0.29, 0.40, 0.47, 0.553], [0.41, 0.43, 0.43, 0.425, 0.415])
    cuerda = np.clip((linea_z - q[:, 2]) / 0.17, 0, 1)
    fw = smooth(0.12, 0.62, cuerda) * smooth(0.51, 0.455, ax)
    f12 = smooth(0.25, 0.33, ax)
    f23 = smooth(0.365, 0.425, ax)
    f = [1 - f12, f12 - f23, f23]
    for b in span:
        W[b][idx] += r[b] * (1 - fw)
    for k in range(3):
        W["feather_%d%s" % (k + 1, s)][idx] += fw * f[k]

# Cuerpo: una sola pieza con torso, cabeza, pico, patas y cola.
q = vco[cuerpo]
ax = np.abs(q[:, 0])
col = ["hips", "spine", "chest", "neck_01", "neck_02", "head"]
# El cuello es corto (8 cm) y en vuelo dobla ~60 grados: transiciones anchas para repartir el giro.
r, _ = cadena(q, [HUESOS[col[0]][1]] + [HUESOS[h][2] for h in col], col, [0.04, 0.05, 0.06, 0.06, 0.05])
# Pico y cara: la base del pico (z 0.51) queda por debajo del hueso de la cabeza (z 0.55), asi que la cadena
# le daba peso de cuello y el pico se doblaba al girar la cabeza. Todo lo que esta delante del cuello va rigido a la cabeza.
CARA = smooth(-0.05, -0.08, q[:, 1]) * smooth(0.48, 0.50, q[:, 2])
for h in col[:-1]:
    r[h] = r[h] * (1 - CARA)
r["head"] = r["head"] * (1 - CARA) + CARA

s_cola = (q - BASE_COLA) @ DIR_COLA
f_cola = LARGO_COLA / 0.376          # las medidas de abajo son del modelo anterior, de cola mas larga
peso_cola = smooth(-0.015, 0.05, s_cola) * smooth(0.095, 0.15, q[:, 1]) * smooth(0.36, 0.30, q[:, 2])
c_cola = smooth(0.135 * f_cola, 0.235 * f_cola, s_cola)
ancho_cola = np.maximum(0.05, np.interp(s_cola, [0.0, 0.08, 0.2, 0.36], [0.09, 0.115, 0.135, 0.135]))
lat = q[:, 0] / ancho_cola
abanico = smooth(0.25, 0.85, np.abs(lat)) * smooth(0.04, 0.16, s_cola)

nombres = col + ["tail_01", "tail_02", "tail_fan.L", "tail_fan.R",
                 "leg.L", "leg.R", "foot.L", "foot.R"]
nombres += ["toe_%s%s" % (d, s) for d in DEDOS for s in (".L", ".R")]
nombres += ["claw_%s%s" % (d, s) for d in DEDOS for s in (".L", ".R")]
pos = {b: i for i, b in enumerate(nombres)}
B = np.zeros((len(q), len(nombres)))
for h in col:
    B[:, pos[h]] = r[h] * (1 - peso_cola)
B[:, pos["tail_01"]] = peso_cola * (1 - c_cola) * (1 - abanico)
B[:, pos["tail_02"]] = peso_cola * c_cola * (1 - abanico)
B[:, pos["tail_fan.L"]] = peso_cola * abanico * (lat > 0)
B[:, pos["tail_fan.R"]] = peso_cola * abanico * (lat <= 0)

# Pierna: muslo que nace del cuerpo, tarso y pie. Lo que no es cola y queda bajo el muslo es pata.
es_cola = (q[:, 1] > 0.15)
libre = ~(r["head"] > 0.999)
fijo_pata = np.zeros(len(q), bool)
for s, k in ((".L", 1.0), (".R", -1.0)):
    lado = (q[:, 0] * k > 0) & ~es_cola
    dxy = np.hypot(q[:, 0] * k - 0.085, q[:, 1] + 0.005)
    pierna = lado * smooth(0.205, 0.15, q[:, 2]) * smooth(0.098, 0.072, dxy)
    pie = lado * smooth(0.112, 0.092, q[:, 2])
    pierna = np.maximum(pierna, pie)
    B[pierna > 0] *= (1 - pierna[pierna > 0])[:, None]
    B[:, pos["leg" + s]] += pierna * (1 - pie)
    B[:, pos["foot" + s]] += pie
    # Dedos: cada vertice del pie va al dedo de cuyo eje esta mas cerca.
    planta = lado & (q[:, 2] < 0.085)
    pxy = np.stack((q[:, 0] * k, q[:, 1]), axis=1) - CENTRO_PIE
    rad = np.linalg.norm(pxy, axis=1)
    claves = list(DEDOS)
    dist = []
    for d in claves:
        u = np.array(DEDOS[d]) - CENTRO_PIE
        u = u / np.linalg.norm(u)
        t = np.clip(pxy @ u, 0, None)
        dist.append(np.linalg.norm(pxy - np.outer(t, u), axis=1))
    cual = np.argmin(np.stack(dist, axis=1), axis=1)
    dedo = planta * smooth(0.022, 0.04, rad)
    punta = smooth(0.075, 0.098, rad)
    for j, d in enumerate(claves):
        m = (cual == j) & (dedo > 0)
        B[np.where(m)[0]] *= (1 - dedo[m])[:, None]
        B[m, pos["toe_%s%s" % (d, s)]] += dedo[m] * (1 - punta[m])
        B[m, pos["claw_%s%s" % (d, s)]] += dedo[m] * punta[m]
    fijo_pata |= planta | (lado & (q[:, 2] < 0.13))

# Suaviza sobre la malla el torso, el cuello y la union de muslos y cola.
# La cabeza, las patas y la punta de la cola quedan fijas para que no se deformen.
local = {int(v): i for i, v in enumerate(cuerpo)}
aristas = np.array([(local[a], local[b]) for a, b in (e.vertices for e in me.edges) if a in local])
fija = ~libre | fijo_pata | (s_cola > 0.10) & (q[:, 1] > 0.15)
B_fija = B[fija].copy()
grado = np.zeros(len(q))
np.add.at(grado, aristas[:, 0], 1); np.add.at(grado, aristas[:, 1], 1)
for _ in range(30):
    suma = np.zeros_like(B)
    np.add.at(suma, aristas[:, 0], B[aristas[:, 1]])
    np.add.at(suma, aristas[:, 1], B[aristas[:, 0]])
    B = 0.5 * B + 0.5 * suma / grado[:, None]
    B[fija] = B_fija
for b, i in pos.items():
    W[b][cuerpo] += B[:, i]

# Penacho y cejas salen del peso de la cabeza.
ph = W["head"][cuerpo].copy()
crest = smooth(0.01, 0.10, q[:, 1]) * smooth(0.53, 0.60, q[:, 2])
W["crest"][cuerpo] += ph * crest
ph *= 1 - crest
for s, k in ((".L", 1.0), (".R", -1.0)):
    pts = [(0.022, -0.128, 0.628), (0.07, -0.112, 0.652), (0.118, -0.062, 0.622)]
    r_b, d_b = cadena(q * np.array((k, 1, 1)), pts, ["a", "b"], [0.02])
    ceja = smooth(0.036, 0.016, d_b) * (q[:, 0] * k > 0)
    W["brow" + s][cuerpo] += ph * ceja
    ph *= 1 - ceja

W["head"][cuerpo] = ph
W["jaw"][pico_bajo] += 1      # mandibula: pieza aparte, rigida

for s in (".L", ".R"):
    W["eye" + s][ojo[s]] += 1

# Las motas son rigidas y copian el peso del vertice de piel mas cercano.
piel = np.where((lab == i_cuerpo) | (lab == i_a) | (lab == i_b))[0]
M = np.stack([W[b] for b in HUESOS], axis=1)
for isl in adornos:
    idx = np.where(lab == isl)[0]
    c = vco[idx].mean(0)
    cerca = piel[np.argmin(np.linalg.norm(vco[piel] - c, axis=1))]
    M[idx] = M[cerca]

# Unity usa 4 influencias por vertice por defecto: se conservan las 4 mayores y se renormaliza.
orden = np.argsort(M, axis=1)
np.put_along_axis(M, orden[:, :-4], 0.0, axis=1)
M[M < 0.01] = 0.0
total = M.sum(axis=1, keepdims=True)
assert (total > 0).all(), "hay vertices sin peso"
M /= total
for j, nombre in enumerate(HUESOS):
    idx = np.where(M[:, j] > 1e-4)[0]
    if len(idx):
        g = obj.vertex_groups.new(name=nombre)
        for i in idx:
            g.add([int(i)], float(M[i, j]), "REPLACE")

obj.parent = arm
mod = obj.modifiers.new("Armature", "ARMATURE")
mod.object = arm

bm = bmesh.new(); bm.from_mesh(me)
print("MALLA vertices", len(bm.verts), "caras", len(bm.faces), "triangulos", sum(len(f.verts) - 2 for f in bm.faces),
      "bordes abiertos", sum(1 for e in bm.edges if e.is_boundary))
bm.free()

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
