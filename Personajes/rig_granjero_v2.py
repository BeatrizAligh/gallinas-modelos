"""
Rig v2 y animaciones de un granjero (chibi en pose T, mirando a -Y, Z arriba). Se ejecuta con Blender por CLI:
blender -b -P rig_granjero_v2.py -- <retopo.fbx> <textura.png|-> <carpeta_salida> <Nombre> [<alta.fbx para hornear>]

Cambios respecto a la v1:
- Esqueleto: columna de 3 huesos, cuello de 2, clavícula con ritmo escápulo-humeral automático, antebrazo con hueso de giro
  (reparte la pronación: sin efecto caramelo), mano con bloque de dedos (2 falanges) y pulgar (2), raíz/cadera y piernas igual.
- Brazos resueltos con la bisagra real del codo (no la rotación mínima): codos que doblan hacia donde deben y palmas controladas.
- Pesos: proxy de una sola pieza (remallado en vóxeles) con calor de huesos y transferencia a la malla real, así mangas,
  guantes y cabeza (piezas sueltas) se deforman igual; las manos (manopla) por distancia a pulgar/dedos.
- Animaciones con anticipación, superposición (brazos y cabeza con retraso), arcos, rebote al asentarse y dedos que agarran.
Brazos (revisión 0.7.1, personaje por personaje):
- Hombro, codo y muñeca medidos con cortes del tubo del brazo (medidas_brazo.py): el hombro va en la raíz de la manga
  (antes quedaba casi en el puño y la manga entera se pegaba al pecho), la muñeca donde acaba el puño (AJUSTES por personaje).
- Bisagra anatómica: el codo dobla hacia delante desde la pose T aunque la palma mire abajo; con el brazo recto se elige el
  giro del húmero más natural y la palma la orienta el antebrazo (pronación ±100°, repartida 55 % a media altura).
- Ayudantes de volumen Deltoid (mitad de la elevación del brazo) y Elbow (mitad entre brazo y antebrazo) + pesos a lo largo
  de la cadena con bandas en cada articulación: al doblar no se aplasta ni se retuerce.
- Piezas sueltas de pelo rígidas con la cabeza; manos sólo a su altura (una falda ancha no se va con las manos).
- El script avisa de clips que piden un codo al revés (¡CODO AL REVÉS!) y da el giro máximo por clip.
Mismos nombres de clips y de huesos clave (Hips, Spine, Grip.L/R, Foot/Toe) que la v1: el juego no cambia.
"""
import bpy, bmesh, sys, os, math
import numpy as np
from mathutils import Vector, Matrix, Quaternion, kdtree

A = sys.argv[sys.argv.index('--') + 1:]
SRC, TEX, OUT, NAME = A[0], A[1], A[2], A[3]
ALTA = A[4] if len(A) > 4 else None
os.makedirs(OUT, exist_ok=True)
FPS = 30

# ============================================================ 1. malla
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=SRC)
body = [o for o in bpy.data.objects if o.type == 'MESH'][0]
body.data.transform(body.matrix_world)
body.matrix_world = Matrix.Identity(4)
body.name = body.data.name = NAME
for o in list(bpy.data.objects):
    if o != body: bpy.data.objects.remove(o)
sc = bpy.context.scene
sc.render.fps = FPS


def select_only(o):
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active = o


# ============================================================ 2. textura (o horneado)
tex_out = os.path.join(OUT, NAME + '.png')
if len(body.data.uv_layers) == 0 and ALTA:
    print('BAKE: sin UVs, UV automático y horneado desde', ALTA)
    select_only(body)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004)
    bpy.ops.object.mode_set(mode='OBJECT')
    img = bpy.data.images.new(NAME + '_bake', 2048, 2048)
    mat = bpy.data.materials.new('Bake'); mat.use_nodes = True
    node = mat.node_tree.nodes.new('ShaderNodeTexImage'); node.image = img
    mat.node_tree.nodes.active = node
    body.data.materials.clear(); body.data.materials.append(mat)
    bpy.ops.import_scene.fbx(filepath=ALTA)
    high = [o for o in bpy.data.objects if o.type == 'MESH' and o != body][0]
    high.data.transform(high.matrix_world); high.matrix_world = Matrix.Identity(4)
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 4
    sc.cycles.device = 'CPU'
    bpy.ops.object.select_all(action='DESELECT')
    high.select_set(True); body.select_set(True)
    bpy.context.view_layer.objects.active = body
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = 0.02
    sc.render.bake.max_ray_distance = 0.06
    sc.render.bake.margin = 8
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
    img.filepath_raw = tex_out; img.file_format = 'PNG'; img.save()
    bpy.data.objects.remove(high)
    body.data.materials.clear()
    print('BAKE: hecho', tex_out)
elif TEX != '-':
    import shutil
    shutil.copyfile(TEX, tex_out)

# ============================================================ 3. medidas
V = np.array([v.co[:] for v in body.data.vertices])
x, y, z = V[:, 0], V[:, 1], V[:, 2]
H = z.max()
bins = np.arange(0, H, 0.005)
span = np.array([np.abs(x[(z >= b) & (z < b + 0.005)]).max() if np.any((z >= b) & (z < b + 0.005)) else 0 for b in bins])
smax = span.max()
arm_bins = bins[span > 0.72 * smax]
armZ = float(np.median(arm_bins)); armTop = float(arm_bins.max()); armBot = float(arm_bins.min())


def half_width(z0, z1, xlim=None):
    m = (z >= z0) & (z < z1)
    if xlim is not None: m &= np.abs(x) < xlim
    return float(np.abs(x[m]).max()) if np.any(m) else 0.0


torso = half_width(armBot - 0.05, armBot - 0.01)
shX = torso * 0.92
tipL, tipR = float(x.max()), float(-x.min())
handLen = 0.24 * ((tipL + tipR) / 2 - shX)
# Cuello: lo más estrecho entre los brazos y la cabeza.
nz = np.arange(armTop + 0.005, min(H, armTop + 0.12 * H), 0.004)
nw = [half_width(a, a + 0.004, torso * 1.1) for a in nz]
neckZ = float(nz[int(np.argmin(nw))]) if len(nz) else armTop + 0.04
if not (0.62 * H <= neckZ <= 0.70 * H): neckZ = 0.66 * H   # pelo/trenzas: lo normal en estos chibis
# Entrepierna: lo más bajo de la columna central (con falda, se queda en un valor razonable).
col = V[(np.abs(x) < 0.015) & (z > 0.05 * H) & (z < 0.6 * H)]
crotch = float(np.clip(col[:, 2].min() if len(col) else 0.36 * H, 0.26 * H, 0.42 * H))
hipZ = crotch + 0.035 * H
feet = V[z < 0.05 * H]
legX = float(np.mean(np.abs(feet[:, 0]))) if len(feet) else 0.08
ankleZ = 0.065 * H
kneeZ = (hipZ + ankleZ) / 2 + 0.01 * H
toeY = float(feet[:, 1].min()) if len(feet) else -0.1
backY = float(np.median(V[(z > armBot - 0.1) & (z < armBot) & (np.abs(x) < torso * 0.5)][:, 1])) if True else 0
spineY = 0.5 * (y[(z > hipZ) & (z < armBot) & (np.abs(x) < torso * 0.6)].min() + y[(z > hipZ) & (z < armBot) & (np.abs(x) < torso * 0.6)].max())
print(f'MEDIDAS {NAME}: H={H:.3f} brazos z={armZ:.3f} ({armBot:.3f}-{armTop:.3f}) torso={torso:.3f} manos={tipL:.3f}/{tipR:.3f} '
      f'cuello={neckZ:.3f} cadera={hipZ:.3f} piernas x={legX:.3f} rodilla={kneeZ:.3f} puntera y={toeY:.3f}')

# ============================================================ 3b. brazos y manos (cortes del tubo del brazo, medidas_brazo.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medidas_brazo import medir_brazo, AJUSTES
# Arreglos de la revisión 0.9.3 (Lola nueva), ACTIVADOS POR PERSONAJE en medidas_brazo.AJUSTES para que relanzar a los
# demás granjeros dé exactamente lo mismo que antes hasta revisarlos uno por uno: 'bajo_brazo', 'sisa', 'capas',
# 'sin_choques' (True para activarlo).
_AJ = AJUSTES.get(NAME, {})
FIX = {k: bool(_AJ.get(k, False)) for k in ('bajo_brazo', 'sisa', 'capas', 'sin_choques')}
print('ARREGLOS 0.9.3:', ', '.join(k for k, v in FIX.items() if v) or 'ninguno')
EDG = np.array([e.vertices[:] for e in body.data.edges])
_MB = {s_: medir_brazo(V, EDG, sg_, armZ, H, name=NAME) for s_, sg_ in (('L', 1), ('R', -1))}


def _sym(key):
    a_, b_ = _MB['L'][key], _MB['R'][key]
    return np.array([(abs(a_[0]) + abs(b_[0])) / 2, (a_[1] + b_[1]) / 2, (a_[2] + b_[2]) / 2])


# El personaje es simétrico: medidas promediadas (el pelo o una trenza a un lado no tuercen el esqueleto).
ARM = dict(S=_sym('shoulder'), E=_sym('elbow'), W=_sym('wrist'), T=_sym('tip_c'),
           r_sh=(_MB['L']['r_sh'] + _MB['R']['r_sh']) / 2, r_el=(_MB['L']['r_el'] + _MB['R']['r_el']) / 2,
           r_wr=(_MB['L']['r_wr'] + _MB['R']['r_wr']) / 2)
_tt = [_MB[s_]['thumb_tip'] for s_ in 'LR']
ARM['thumb'] = np.array([(abs(_tt[0][0]) + abs(_tt[1][0])) / 2, (_tt[0][1] + _tt[1][1]) / 2, (_tt[0][2] + _tt[1][2]) / 2])     if _tt[0] is not None and _tt[1] is not None else None
_prof = _MB['L']['prof']


def tube_at(xq):
    i = int(np.argmin(np.abs(_prof[:, 0] - xq)))
    return _prof[i]


shX = float(ARM['S'][0])
tipL = tipR = float(ARM['T'][0])
handLen = float(ARM['T'][0] - ARM['W'][0])
print(f'BRAZO (simétrico): hombro {np.round(ARM["S"], 3)} codo {np.round(ARM["E"], 3)} muñeca {np.round(ARM["W"], 3)} '
      f'punta {np.round(ARM["T"], 3)} pulgar {None if ARM["thumb"] is None else np.round(ARM["thumb"], 3)} '
      f'radios {ARM["r_sh"]:.3f}/{ARM["r_el"]:.3f}/{ARM["r_wr"]:.3f}')


def hand_info(sgn):
    W_, T_ = ARM['W'], ARM['T']
    hc = tube_at(W_[0] + 0.45 * handLen)
    cy, cz = float(hc[1]), float(hc[2])
    if ARM['thumb'] is None:
        return dict(wrist=float(W_[0]), tip=float(T_[0]), cy=cy, cz=cz, thumb_dir=Vector((0, -1, 0)), thumb_base=None, thumb_tip=None)
    tt = ARM['thumb']
    tc = tube_at(tt[0])
    td = Vector((0, tt[1] - tc[1], tt[2] - tc[2])).normalized()
    rh = float((tc[3] + tc[4]) / 2)
    bx = W_[0] + 0.3 * (tt[0] - W_[0])
    base = Vector((sgn * bx, tc[1], tc[2])) + td * 0.45 * rh
    return dict(wrist=float(W_[0]), tip=float(T_[0]), cy=cy, cz=cz, thumb_dir=td, thumb_base=base,
                thumb_tip=Vector((sgn * tt[0], tt[1], tt[2])))


HAND = {'L': hand_info(1), 'R': hand_info(-1)}
for s_ in 'LR':
    td = HAND[s_]['thumb_dir']
    print(f'MANO {s_}: pulgar hacia {"arriba" if abs(td.z) >= abs(td.y) else "delante"} {tuple(round(c, 2) for c in td)}')

# ============================================================ 4. esqueleto v2
arm_obj = bpy.data.objects.new('Granjero', bpy.data.armatures.new('Granjero'))
sc.collection.objects.link(arm_obj)
select_only(arm_obj)
bpy.ops.object.mode_set(mode='EDIT')
eb = arm_obj.data.edit_bones
J = {}


def bone(name, head, tail, parent=None, deform=True, connect=False, roll=0.0):
    b = eb.new(name)
    b.head, b.tail = Vector(head), Vector(tail)
    b.roll = roll
    if parent: b.parent = eb[parent]; b.use_connect = connect
    b.use_deform = deform
    J[name] = (Vector(head), Vector(tail))
    return b


sy = spineY
bone('Root', (0, sy, 0), (0, sy, 0.15 * H), deform=False)
bone('Hips', (0, sy, hipZ), (0, sy, hipZ + 0.06 * H), 'Root')
s0 = hipZ + 0.06 * H
s3 = neckZ - 0.01 * H
s1 = s0 + (s3 - s0) * 0.34
s2 = s0 + (s3 - s0) * 0.67
bone('Spine', (0, sy, s0), (0, sy, s1), 'Hips', connect=True)
bone('Spine1', (0, sy, s1), (0, sy, s2), 'Spine', connect=True)
bone('Chest', (0, sy, s2), (0, sy, s3), 'Spine1', connect=True)
n1 = neckZ + 0.012 * H
bone('Neck', (0, sy, s3), (0, sy, n1), 'Chest', connect=True)
bone('Neck1', (0, sy, n1), (0, sy, neckZ + 0.035 * H), 'Neck', connect=True)
bone('Head', (0, sy, neckZ + 0.035 * H), (0, sy, H * 0.98), 'Neck1', connect=True)
for side, sgn in (('L', 1), ('R', -1)):
    hi = HAND[side]
    wrist, tip = hi['wrist'], hi['tip']
    m3 = lambda p_: (sgn * float(p_[0]), float(p_[1]), float(p_[2]))
    Sx, Ex, Wx = ARM['S'], ARM['E'], ARM['W']
    Mx = Ex + 0.5 * (Wx - Ex)
    knuck = wrist + 0.5 * handLen
    f2 = wrist + 0.76 * handLen
    hz, hy = hi['cz'], hi['cy']
    # Clavícula: desde cerca del cuello hasta la articulación del hombro (dentro de la manga, a la altura de la axila).
    bone(f'Shoulder.{side}', (sgn * 0.25 * Sx[0], sy, Sx[2] + 0.01 * H), m3(Sx), 'Chest')
    bone(f'UpperArm.{side}', m3(Sx), m3(Ex), f'Shoulder.{side}', connect=True)
    bone(f'LowerArm.{side}', m3(Ex), m3(Mx), f'UpperArm.{side}', connect=True)
    bone(f'LowerArmTwist.{side}', m3(Mx), m3(Wx), f'LowerArm.{side}', connect=True)
    bone(f'Hand.{side}', m3(Wx), (sgn * knuck, hy, hz), f'LowerArmTwist.{side}', connect=True)
    bone(f'Fingers1.{side}', (sgn * knuck, hy, hz), (sgn * f2, hy, hz), f'Hand.{side}', connect=True)
    bone(f'Fingers2.{side}', (sgn * f2, hy, hz), (sgn * tip, hy, hz), f'Fingers1.{side}', connect=True)
    # Ayudantes de volumen (deforman, con rotación intermedia): codo y hombro no se aplastan al doblar.
    bone(f'Elbow.{side}', m3(Ex), m3(Ex + 0.3 * (Wx - Ex)), f'UpperArm.{side}')
    bone(f'Deltoid.{side}', m3(Sx), m3(Sx + 0.35 * (Ex - Sx)), f'Shoulder.{side}')
    if hi['thumb_base'] is not None:
        tb, tt = hi['thumb_base'], hi['thumb_tip']
        tm = tb.lerp(tt, 0.5)
        bone(f'Thumb1.{side}', tb[:], tm[:], f'Hand.{side}')
        bone(f'Thumb2.{side}', tm[:], tt[:], f'Thumb1.{side}', connect=True)
    # Agarre de las herramientas: en el centro del puño, apuntando arriba en la pose T (no deforma).
    palm = wrist + 0.45 * handLen
    bone(f'Grip.{side}', (sgn * palm, hy, hz), (sgn * palm, hy, hz + 0.06 * H), f'Hand.{side}', deform=False)
    bone(f'UpperLeg.{side}', (sgn * legX, sy, hipZ), (sgn * legX, sy, kneeZ), 'Hips')
    bone(f'LowerLeg.{side}', (sgn * legX, sy, kneeZ), (sgn * legX, sy, ankleZ), f'UpperLeg.{side}', connect=True)
    bone(f'Foot.{side}', (sgn * legX, sy, ankleZ), (sgn * legX, toeY * 0.55 + sy * 0.45, 0.02 * H), f'LowerLeg.{side}', connect=True)
    bone(f'Toe.{side}', (sgn * legX, toeY * 0.55 + sy * 0.45, 0.02 * H), (sgn * legX, toeY, 0.015 * H), f'Foot.{side}', connect=True)
bpy.ops.object.mode_set(mode='OBJECT')

# ============================================================ 5. pesos v2 (proxy de una pieza → transferencia)
DEFORM = [b.name for b in arm_obj.data.bones if b.use_deform]


def distance_weights(obj, names, mask=None, power=4.0, top=2):
    """Pesos por distancia a los segmentos de los huesos (sin cruzar de lado). mask: índices de vértices a tocar."""
    segs = [(n, np.array(arm_obj.data.bones[n].head_local[:]), np.array(arm_obj.data.bones[n].tail_local[:])) for n in names]
    co = np.array([v.co[:] for v in obj.data.vertices])
    idx = np.arange(len(co)) if mask is None else np.asarray(mask)
    c = co[idx]
    dist = np.zeros((len(c), len(segs)))
    for j, (n, h, t) in enumerate(segs):
        ab = t - h
        u = np.clip(((c - h) @ ab) / max(1e-9, ab @ ab), 0, 1)
        d = np.linalg.norm(c - (h + u[:, None] * ab), axis=1)
        side = 1 if n.endswith('.L') else -1 if n.endswith('.R') else 0
        if side: d = d + np.where(c[:, 0] * side < -0.01, 1.0, 0.0)
        dist[:, j] = d
    vgs = {n: (obj.vertex_groups.get(n) or obj.vertex_groups.new(name=n)) for n, _, _ in segs}
    allg = list(obj.vertex_groups)
    for k, i in enumerate(idx):
        for g in allg:
            try: g.remove([int(i)])
            except RuntimeError: pass
        order = np.argsort(dist[k])[:top]
        w = 1.0 / np.maximum(dist[k][order], 1e-3) ** power
        w /= w.sum()
        for j, wj in zip(order, w):
            if wj > 0.02: vgs[segs[j][0]].add([int(i)], float(wj), 'REPLACE')


# Proxy: remallado en vóxeles (una sola pieza, cerrada) y aligerado: el calor de huesos funciona casi siempre.
# Si falla con un tamaño de vóxel, se prueba con otros antes de recurrir a los pesos por distancia.
def make_proxy(voxel, target):
    pr = body.copy(); pr.data = body.data.copy(); pr.name = 'Proxy'
    sc.collection.objects.link(pr)
    for vg in list(pr.vertex_groups): pr.vertex_groups.remove(vg)
    rm = pr.modifiers.new('Remesh', 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = voxel
    select_only(pr); bpy.ops.object.modifier_apply(modifier='Remesh')
    if len(pr.data.polygons) > target:
        dm = pr.modifiers.new('Dec', 'DECIMATE'); dm.ratio = target / len(pr.data.polygons)
        bpy.ops.object.modifier_apply(modifier='Dec')
    select_only(pr); arm_obj.select_set(True); bpy.context.view_layer.objects.active = arm_obj
    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    except Exception as e:
        print('PESOS: calor en el proxy falló:', e)
    ok = sum(1 for v in pr.data.vertices if sum(g.weight for g in v.groups) > 0.01)
    return pr, ok


proxy = None
for voxel, target in ((H / 95.0, 40000), (H / 80.0, 26000), (H / 120.0, 30000), (H / 65.0, 16000)):
    if proxy is not None: bpy.data.objects.remove(proxy)
    proxy, pw = make_proxy(voxel, target)
    print(f'PROXY: vóxel {voxel:.4f}, {len(proxy.data.vertices)} vértices, {pw} con peso')
    if pw >= 0.8 * len(proxy.data.vertices): break
else:
    print(f'PESOS: calor insuficiente en el proxy ({pw}), por distancia')
    distance_weights(proxy, DEFORM)
proxy.parent = None
for m in list(proxy.modifiers): proxy.modifiers.remove(m)

# Transferencia a la malla real (vecino interpolado en la superficie del proxy): pesos iguales en mangas, guantes, cabeza...
for vg in list(body.vertex_groups): body.vertex_groups.remove(vg)
for n in DEFORM: body.vertex_groups.new(name=n)
dt = body.modifiers.new('DT', 'DATA_TRANSFER')
dt.object = proxy
dt.use_vert_data = True
dt.data_types_verts = {'VGROUP_WEIGHTS'}
dt.vert_mapping = 'POLYINTERP_NEAREST'
dt.layers_vgroup_select_src = 'ALL'
dt.layers_vgroup_select_dst = 'NAME'
select_only(body)
bpy.ops.object.modifier_apply(modifier='DT')
bpy.data.objects.remove(proxy)

# Manos: pulgar, bloque de dedos y mano por distancia (la manopla no la entiende bien el calor).
for side, sgn in (('L', 1), ('R', -1)):
    hi = HAND[side]
    hm = [v.index for v in body.data.vertices if v.co.x * sgn > hi['wrist'] - 0.25 * handLen
          and abs(v.co.z - ARM['W'][2]) < 2.6 * ARM['r_wr'] + 0.03]          # sólo la mano (no el bajo de una falda ancha)
    names = [f'LowerArmTwist.{side}', f'Hand.{side}', f'Fingers1.{side}', f'Fingers2.{side}']
    if f'Thumb1.{side}' in arm_obj.data.bones: names += [f'Thumb1.{side}', f'Thumb2.{side}']
    distance_weights(body, names, hm, power=3.0, top=2)

# Lo que haya quedado sin peso: del vecino más cercano.
weighted = [i for i, v in enumerate(body.data.vertices) if sum(g.weight for g in v.groups) > 0.01]
kd = kdtree.KDTree(len(weighted))
for k, i in enumerate(weighted): kd.insert(body.data.vertices[i].co, k)
kd.balance()
missing = 0
for v in body.data.vertices:
    if sum(g.weight for g in v.groups) > 0.01: continue
    _, k, _ = kd.find(v.co)
    src = body.data.vertices[weighted[k]]
    for g in src.groups: body.vertex_groups[g.group].add([v.index], g.weight, 'REPLACE')
    missing += 1
# Cabeza, pelo y sombrero rígidos con la cabeza (el ala no se dobla); justo bajo la cabeza, fundido con el cuello.
head_vg, neck1_vg = body.vertex_groups['Head'], body.vertex_groups['Neck1']
cut = neckZ + 0.035 * H
band = 0.02 * H
for v in body.data.vertices:
    if v.co.z <= cut - band: continue
    for g in list(v.groups): body.vertex_groups[g.group].remove([v.index])
    w = min(1.0, max(0.0, (v.co.z - (cut - band)) / band))
    if w >= 0.999: head_vg.add([v.index], 1.0, 'REPLACE')
    else:
        head_vg.add([v.index], w, 'REPLACE')
        neck1_vg.add([v.index], 1.0 - w, 'REPLACE')
select_only(body)
try:
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    bpy.ops.object.vertex_group_smooth(group_select_mode='ALL', factor=0.35, repeat=1)
except Exception as e:
    print('PESOS: sin suavizado:', e)
bpy.ops.object.mode_set(mode='OBJECT')


# ---- Piezas sueltas de pelo (moños, trenzas) que bajan hasta los hombros: rígidas con la cabeza, nunca con el brazo.
def islands(me):
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    seen = np.zeros(len(bm.verts), bool); out = []
    for v0 in bm.verts:
        if seen[v0.index]: continue
        st = [v0]; seen[v0.index] = True; ids = []
        while st:
            a_ = st.pop(); ids.append(a_.index)
            for e_ in a_.link_edges:
                b_ = e_.other_vert(a_)
                if not seen[b_.index]: seen[b_.index] = True; st.append(b_)
        out.append(np.array(ids))
    bm.free()
    return sorted(out, key=len, reverse=True)


CO = np.array([v.co[:] for v in body.data.vertices])
ISL = islands(body.data)
head_only = np.zeros(len(CO), bool)
for ids in ISL[1:]:
    P_ = CO[ids]
    if P_[:, 2].max() > cut and P_[:, 2].mean() > ARM['S'][2]:
        head_only[ids] = True
for i in np.where(head_only)[0]:
    v = body.data.vertices[int(i)]
    for g in list(v.groups): body.vertex_groups[g.group].remove([v.index])
    head_vg.add([v.index], 1.0, 'REPLACE')
print(f'PESOS: {int(head_only.sum())} vértices de piezas de pelo rígidos con la cabeza')


# ---- Brazos: pesos a lo largo de la cadena hombro-codo-muñeca (tubo), con bandas en las articulaciones y ayudantes
# de rotación intermedia (Deltoid, Elbow): al doblar, la piel gira como un sólido alrededor de la articulación y no se
# aplasta. El giro del antebrazo se reparte de codo a muñeca (sin efecto caramelo).
def smooth01(a_, b_, x_):
    u_ = np.clip((x_ - a_) / (b_ - a_), 0, 1)
    return u_ * u_ * (3 - 2 * u_)


def keyed(svals, keys):
    """Pesos por tramos (interpolación suave entre claves (s, {hueso: peso}))."""
    names_ = sorted({n for _, d_ in keys for n in d_})
    W_ = np.zeros((len(svals), len(names_)))
    for j_, n in enumerate(names_):
        vals = np.array([d_.get(n, 0.0) for _, d_ in keys]); ss = np.array([k_ for k_, _ in keys])
        u_ = np.interp(svals, ss, np.arange(len(ss)))
        i0 = np.clip(np.floor(u_).astype(int), 0, len(ss) - 2); fr = u_ - i0
        fr = fr * fr * (3 - 2 * fr)
        W_[:, j_] = vals[i0] * (1 - fr) + vals[i0 + 1] * fr
    W_ /= np.maximum(W_.sum(1, keepdims=True), 1e-9)
    return names_, W_


ARM_MASK = np.zeros(len(CO), bool)
ARM_GROUPS = {'Deltoid', 'UpperArm', 'Elbow', 'LowerArm', 'LowerArmTwist', 'Hand', 'Fingers1', 'Fingers2', 'Thumb1', 'Thumb2'}
for side, sgn in (('L', 1), ('R', -1)):
    mir = np.array([sgn, 1, 1])
    S_, E_, W_, T_ = ARM['S'] * mir, ARM['E'] * mir, ARM['W'] * mir, ARM['T'] * mir
    LSE, LEW, LWT = np.linalg.norm(E_ - S_), np.linalg.norm(W_ - E_), np.linalg.norm(T_ - W_)
    cand = np.where((CO[:, 0] * sgn > S_[0] * sgn - 2.2 * ARM['r_sh']) & ~head_only & (CO[:, 2] < cut - band))[0]
    P_ = CO[cand]
    best = np.full(len(P_), 1e9); spar = np.zeros(len(P_))
    for k_, (a_, b_, L_) in enumerate(((S_, E_, LSE), (E_, W_, LEW), (W_, T_, LWT))):
        ab = b_ - a_
        u_ = ((P_ - a_) @ ab) / (L_ * L_)
        uc = np.clip(u_, 0 if k_ else -0.8, 1)          # antes del hombro: se prolonga el eje (s < 0)
        d_ = np.linalg.norm(P_ - (a_ + uc[:, None] * ab), axis=1)
        m_ = d_ < best
        best[m_] = d_[m_]; spar[m_] = k_ + uc[m_]
    r_loc = np.interp(spar, [-0.8, 0, 1, 2, 3], [ARM['r_sh'] * 1.2, ARM['r_sh'], ARM['r_el'], ARM['r_wr'], ARM['r_wr'] * 1.4])
    bS = float(np.clip(0.9 * ARM['r_sh'] / LSE, 0.15, 0.45))
    bEu = float(np.clip(0.85 * ARM['r_el'] / LSE, 0.12, 0.38))
    bEl = float(np.clip(0.85 * ARM['r_el'] / LEW, 0.12, 0.38))
    bW = float(np.clip(0.6 * ARM['r_wr'] / LEW, 0.10, 0.30))
    f_tube = 1 - smooth01(1.45, 2.0, best / np.maximum(r_loc, 1e-3))
    # Pasada la muñeca todo es mano (pose T)... pero sólo a la altura del brazo: una falda ancha llega a esa x más abajo.
    hand_zone = (CO[cand, 0] * sgn > W_[0] * sgn - 0.3 * ARM['r_wr']) & (np.abs(CO[cand, 2] - W_[2]) < 2.6 * ARM['r_wr'] + 0.02) & (spar > 1.85)
    f_tube[hand_zone] = 1.0
    f_s = smooth01(-1.6 * bS, -0.5 * bS, spar)
    f = f_tube * f_s
    # Debajo del brazo en la pose T (costado del torso, bolsillos del chaleco...) no es brazo aunque quede cerca de su eje:
    # con el radio del tubo (hinchado en la axila) el costado y los bolsillos cogían peso de brazo/deltoides y, al subir el
    # brazo, se estiraban en picos y rasgaban el chaleco (revisión 0.9.3, Lola nueva). Borde inferior medido del tubo.
    apit = float(min(_MB['L']['armpit'], _MB['R']['armpit']))
    pr_ = _prof[_prof[:, 0] >= apit]
    zb_pr = pr_[:, 2] - pr_[:, 4]
    zb0 = float(np.median(zb_pr[pr_[:, 0] < apit + 0.04])) if np.any(pr_[:, 0] < apit + 0.04) else float(zb_pr[0])
    zb = np.interp(np.abs(CO[cand, 0]), pr_[:, 0], zb_pr, left=zb0)
    zb = np.where(np.abs(CO[cand, 0]) < apit + 0.04, zb0, zb)
    below = smooth01(0.004, 0.03, zb - CO[cand, 2]) * (spar < 1.4) * ~hand_zone
    if not FIX['bajo_brazo']: below = np.zeros(len(cand))
    f = f * (1 - below)
    ARM_MASK[cand[f > 0.3]] = True
    sd = f'.{side}'
    keys = [(-1.0, {'Shoulder' + sd: 1}), (-bS, {'Shoulder' + sd: 1}), (0.0, {'Deltoid' + sd: 1}),
            (min(0.42, bS + 0.12), {'UpperArm' + sd: 1}), (1 - bEu, {'UpperArm' + sd: 1}), (1.0, {'Elbow' + sd: 1}),
            (1 + bEl, {'LowerArm' + sd: 1}), (1.5, {'LowerArm' + sd: 0.45, 'LowerArmTwist' + sd: 0.55}),
            (2 - bW, {'LowerArmTwist' + sd: 1}), (2 + 0.35 * bW, {'Hand' + sd: 1}), (3.5, {'Hand' + sd: 1})]
    names_, WA = keyed(spar, keys)
    # Mano: reparto mano / dedos / pulgar por distancia a sus huesos (encima de lo anterior pasada la muñeca).
    hand_names = [n for n in ('Hand' + sd, 'Fingers1' + sd, 'Fingers2' + sd, 'Thumb1' + sd, 'Thumb2' + sd) if n in arm_obj.data.bones]
    segs = [(np.array(arm_obj.data.bones[n].head_local[:]), np.array(arm_obj.data.bones[n].tail_local[:])) for n in hand_names]
    DH = np.zeros((len(P_), len(segs)))
    for j_, (h_, t_) in enumerate(segs):
        ab = t_ - h_
        u_ = np.clip(((P_ - h_) @ ab) / (ab @ ab), 0, 1)
        DH[:, j_] = np.linalg.norm(P_ - (h_ + u_[:, None] * ab), axis=1)
    WH = 1.0 / np.maximum(DH, 1e-3) ** 4
    top2 = np.argsort(DH, 1)[:, :2]
    mask2 = np.zeros_like(WH, bool); np.put_along_axis(mask2, top2, True, 1)
    WH = np.where(mask2, WH, 0); WH /= WH.sum(1, keepdims=True)
    h_mix = smooth01(2.0, 2.0 + 0.8 * bW, spar)
    allnames = sorted(set(names_) | set(hand_names))
    WF = np.zeros((len(P_), len(allnames)))
    for j_, n in enumerate(names_): WF[:, allnames.index(n)] += WA[:, j_] * (1 - h_mix)
    for j_, n in enumerate(hand_names): WF[:, allnames.index(n)] += WH[:, j_] * h_mix
    for n in allnames:
        if n not in body.vertex_groups: body.vertex_groups.new(name=n)
    gidx = {g.index: g.name for g in body.vertex_groups}
    nset = 0
    for k_, i in enumerate(cand):
        if f[k_] < 0.002 and below[k_] < 0.002: continue
        v = body.data.vertices[int(i)]
        old = {gidx[g.group]: g.weight for g in v.groups}
        if below[k_] > 0.002:                    # bajo el brazo: fuera también el peso de brazo que trajo el proxy
            # El peso retirado pasa EXPLÍCITAMENTE al tronco (si sólo hubiera grupos de brazo, normalizar lo devolvería).
            quit_ = sum(w_ * below[k_] for n, w_ in old.items() if n.endswith(sd) and n.split('.')[0] in ARM_GROUPS)
            old = {n: (w_ * (1 - below[k_]) if n.endswith(sd) and n.split('.')[0] in ARM_GROUPS else w_) for n, w_ in old.items()}
            if quit_ > 0:
                tronco = 'Chest' if CO[i, 2] > (s1 + s2) / 2 else 'Spine1'
                old[tronco] = old.get(tronco, 0.0) + quit_
        tot = sum(old.values()) or 1.0
        new_w = {n: w_ / tot * (1 - f[k_]) for n, w_ in old.items()}
        for j_, n in enumerate(allnames):
            if WF[k_, j_] > 1e-4: new_w[n] = new_w.get(n, 0.0) + WF[k_, j_] * f[k_]
        for g in list(v.groups): body.vertex_groups[g.group].remove([v.index])
        for n, w_ in new_w.items():
            if w_ > 0.005: body.vertex_groups[n].add([v.index], float(w_), 'REPLACE')
        nset += 1
    print(f'PESOS brazo {side}: {nset} vértices por cadena (bandas hombro {bS:.2f}, codo {bEu:.2f}/{bEl:.2f}, muñeca {bW:.2f})')



# ---- Capas de ropa (revisión 0.9.3, Lola nueva: chaleco encima de la camisa en la misma pieza). La capa de dentro copia
# los pesos de la de fuera en el punto que tiene justo encima (rayo por su normal): si no, al subir el brazo la camisa,
# que está más cerca del eje del brazo y recibe más peso de la cadena, atraviesa el chaleco.
def capas_de_ropa(dist=0.02 * H, banda=0.012):
    from mathutils.bvhtree import BVHTree
    from mathutils.interpolate import poly_3d_calc
    bm_ = bmesh.new(); bm_.from_mesh(body.data)
    bm_.verts.ensure_lookup_table(); bm_.faces.ensure_lookup_table()
    bvh_ = BVHTree.FromBMesh(bm_)
    gname = {g.index: g.name for g in body.vertex_groups}
    Wd = [{gname[g.group]: g.weight for g in v.groups} for v in body.data.vertices]
    pares = []
    for v in bm_.verts:
        i = v.index
        if head_only[i] or v.co.z > cut - band: continue
        n_ = v.normal
        hit, hn, fi, d_ = bvh_.ray_cast(v.co + n_ * 1e-4, n_, dist)
        if hit is None or hn.dot(n_) < 0.3: continue
        ids = [u.index for u in bm_.faces[fi].verts]
        if i in ids or any(head_only[j] for j in ids): continue
        pares.append((i, ids, poly_3d_calc([bm_.verts[j].co for j in ids], hit)))
    for _ in range(3):                                   # capas de 3 (pañuelo sobre chaleco sobre camisa)
        new = {}
        for i, ids, bw in pares:
            acc = {}
            for j, b_ in zip(ids, bw):
                for n_, w_ in Wd[j].items(): acc[n_] = acc.get(n_, 0.0) + w_ * b_
            new[i] = acc
        for i, acc in new.items(): Wd[i] = acc
    # Borde: la camisa que asoma justo fuera del chaleco (sisa) se funde con la copiada en `banda` metros.
    tapados = np.zeros(len(CO), bool)
    for i, _, _ in pares: tapados[i] = True
    if tapados.any():
        kt = kdtree.KDTree(int(tapados.sum()))
        for i in np.where(tapados)[0]: kt.insert(CO[i], int(i))
        kt.balance()
        for v in bm_.verts:
            i = v.index
            if tapados[i] or head_only[i] or v.co.z > cut - band: continue
            co_, j, d_ = kt.find(v.co)
            if d_ >= banda: continue
            # sólo si el vecino tapado está en la misma superficie (no salta del chaleco a la camisa de al lado)
            if v.normal.dot(bm_.verts[j].normal) < 0.5: continue
            k_ = 0.5 * (1 - d_ / banda)
            acc = {n_: w_ * (1 - k_) for n_, w_ in Wd[i].items()}
            for n_, w_ in Wd[j].items(): acc[n_] = acc.get(n_, 0.0) + w_ * k_
            Wd[i] = acc
    toc = set(i for i, _, _ in pares)
    for i, acc in enumerate(Wd):
        v = body.data.vertices[i]
        old = {gname[g.group]: g.weight for g in v.groups}
        if old == acc: continue
        for g in list(v.groups): body.vertex_groups[g.group].remove([i])
        tot = sum(acc.values()) or 1.0
        for n_, w_ in acc.items():
            if w_ / tot > 0.005: body.vertex_groups[n_].add([i], float(w_ / tot), 'REPLACE')
    bm_.free()
    print(f'PESOS capas: {len(toc)} vértices de capas interiores copian los pesos de la capa de encima')


# ---- Sisa (revisión 0.9.3): alrededor de la axila, el borde del chaleco y la manga que sale de debajo tenían saltos de
# peso (deltoides / brazo) en pocos milímetros y, con el brazo muy arriba (NetSwing, Hammer, Throw), el chaleco se rasgaba
# y la manga se doblaba por encima. Suavizado espacial (por distancia, no por aristas: capas y bordes sueltos reciben lo
# mismo) sólo en la zona de la sisa.
def suavizar_sisa(radio=0.028, pasadas=3):
    if pasadas <= 0: return
    gname = {g.index: g.name for g in body.vertex_groups}
    Wd = [{gname[g.group]: g.weight for g in v.groups} for v in body.data.vertices]
    S_ = ARM['S']; LSE_ = float(np.linalg.norm(ARM['E'] - ARM['S']))
    zona = (~head_only) & (np.abs(CO[:, 0]) > S_[0] - 0.035) & (np.abs(CO[:, 0]) < S_[0] + 0.5 * LSE_)         & (CO[:, 2] > S_[2] - 0.11) & (CO[:, 2] < S_[2] + 0.08)
    ids = np.where(zona)[0]
    kd_ = kdtree.KDTree(len(CO))
    for i, c in enumerate(CO):
        if not head_only[i]: kd_.insert(c, i)
    kd_.balance()
    vec = {int(i): [(j, math.exp(-(d / (0.5 * radio)) ** 2)) for _, j, d in kd_.find_range(CO[i], radio)
                    if (CO[j, 0] > 0) == (CO[i, 0] > 0)] for i in ids}
    for _ in range(pasadas):
        new = {}
        for i, nb in vec.items():
            acc, tw = {}, 0.0
            for j, w in nb:
                tw += w
                for n_, w_ in Wd[j].items(): acc[n_] = acc.get(n_, 0.0) + w_ * w
            new[i] = {n_: w_ / tw for n_, w_ in acc.items()}
        for i, a in new.items(): Wd[i] = a
    for i in ids:
        v = body.data.vertices[int(i)]
        for g in list(v.groups): body.vertex_groups[g.group].remove([v.index])
        tot = sum(Wd[i].values()) or 1.0
        for n_, w_ in Wd[i].items():
            if w_ / tot > 0.005: body.vertex_groups[n_].add([v.index], float(w_ / tot), 'REPLACE')
    print(f'PESOS sisa: {len(ids)} vértices suavizados por distancia ({pasadas} pasadas, radio {radio:.3f})')


if FIX['sisa']: suavizar_sisa()
if FIX['capas']: capas_de_ropa()
bpy.ops.object.vertex_group_limit_total(limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)
print(f'PESOS v2: transferidos desde el proxy, {missing} rellenados por vecindad')

mod = body.modifiers.new('Armature', 'ARMATURE'); mod.object = arm_obj
body.parent = arm_obj

# ============================================================ 6. solver v2 (bisagras reales + clavícula automática)
pb = arm_obj.pose.bones
REST = {b.name: b.matrix_local.copy() for b in arm_obj.data.bones}
FWD, RIGHT, UP = Vector((0, -1, 0)), Vector((-1, 0, 0)), Vector((0, 0, 1))
PITCH, YAW, ROLL = Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, -1, 0))   # + = adelante, izquierda, inclinar a la derecha


def q(axis, deg):
    return Quaternion(axis, math.radians(deg))


def D(f=0.0, r=0.0, u=0.0):
    """Dirección en el cuerpo: adelante, derecha (del personaje), arriba."""
    return (FWD * f + RIGHT * r + UP * u).normalized()


def aim(bone_name, direction):
    h, t = J[bone_name]
    return (t - h).normalized().rotation_difference(direction)


def frame_rot(a1, b1, a2, b2):
    """Rotación que lleva el par (a1, b1) al par (a2, b2) (b se ortogonaliza respecto a a)."""
    def basis(a, b):
        a = a.normalized()
        b = (b - a * a.dot(b))
        if b.length < 1e-6: b = a.orthogonal()
        b.normalize()
        return Matrix((a, b, a.cross(b))).transposed()
    return (basis(a2, b2) @ basis(a1, b1).transposed()).to_quaternion()


def solve(abs_rot, hips_offset=Vector()):
    posed = {}
    for b in arm_obj.data.bones:
        R = abs_rot.get(b.name)
        if R is None:
            R = posed_rot.get(b.parent.name, Quaternion()) if b.parent else Quaternion()
        posed_rot[b.name] = R
        rest = REST[b.name]
        if b.parent:
            pr = REST[b.parent.name]
            head = posed[b.parent.name] @ (pr.inverted() @ rest.translation)
        else:
            head = rest.translation.copy()
        if b.name == 'Hips': head = head + hips_offset
        M = Matrix.Translation(head) @ R.to_matrix().to_4x4() @ rest.to_3x3().to_4x4()
        posed[b.name] = M
        local = rest if not b.parent else REST[b.parent.name].inverted() @ rest
        parent_pose = posed[b.parent.name] if b.parent else Matrix.Identity(4)
        pb[b.name].matrix_basis = local.inverted() @ parent_pose.inverted() @ M
    return posed


posed_rot = {}

# Marco de reposo de cada brazo: dirección del brazo, palma (según hacia dónde mire el pulgar) y bisagra del codo.
ARM_REST = {}
BONE_DIR = {}
for s_, sg in (('L', 1), ('R', -1)):
    d = Vector((sg, 0, 0))
    td = HAND[s_]['thumb_dir']
    palm = Vector((0, -1, 0)) if abs(td.z) >= abs(td.y) else Vector((0, 0, -1))
    # Bisagra anatómica del codo: el antebrazo se dobla HACIA DELANTE desde la pose T, mire donde mire la palma
    # (estos chibis tienen la palma hacia abajo y el pulgar delante: antebrazo ya en pronación).
    ARM_REST[s_] = (d, palm, d.cross(FWD).normalized())
    # Dirección de reposo real de cada hueso (el brazo de la malla no está perfectamente recto en x).
    for n in ('UpperArm', 'LowerArm'):
        h_, t_ = J[f'{n}.{s_}']
        BONE_DIR[f'{n}.{s_}'] = (t_ - h_).normalized()


def qslerp(a, b, k):
    if a.dot(b) < 0: b = Quaternion((-b.w, -b.x, -b.y, -b.z))
    return a.slerp(b, k)


TWIST_STATS = {}
WARN = {}


def side_vec(v, side):
    """Espejo izquierda/derecha de una dirección pensada para el brazo derecho."""
    return v if side == 'R' else Vector((-v.x, v.y, v.z))


def body_frame(pitch=0.0, yaw=0.0, roll=0.0):
    return q(YAW, yaw) @ q(PITCH, pitch) @ q(ROLL, roll)


def scale3(t, k):
    return tuple(a * k for a in t)


def add3(a, b):
    return tuple(x + y for x, y in zip(a, b))


class Pose:
    def __init__(self):
        self.rot = {}
        self.hips = Vector()

    def torso(self, hips=(0, 0, 0), spine=(0, 0, 0), chest=(0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0)):
        """La curva de la espalda se reparte en 3 huesos y el giro de cabeza en 2 de cuello + cabeza (más natural)."""
        bh = body_frame(*hips)
        bs = bh @ body_frame(*spine)
        bs1 = bs @ body_frame(*scale3(chest, 0.45))
        bc = bs1 @ body_frame(*scale3(chest, 0.55))
        bn = bc @ body_frame(*add3(scale3(neck, 0.5), scale3(head, 0.2)))
        bn1 = bn @ body_frame(*add3(scale3(neck, 0.5), scale3(head, 0.25)))
        bhd = bn1 @ body_frame(*scale3(head, 0.55))
        self.rot.update(Hips=bh, Spine=bs, Spine1=bs1, Chest=bc, Neck=bn, Neck1=bn1, Head=bhd, Root=Quaternion())
        self.chest_q, self.hips_q = bc, bh
        return self

    def arm(self, side, upper, lower, palm=None, curl=0.3, wrist=0.0, shrug=0.0, protract=0.0, thumb=None):
        """Brazo con bisagra de codo real. Direcciones del lado derecho en el marco del pecho (se espejan para el izquierdo).
        palm: hacia dónde mira la palma (por defecto, hacia el cuerpo y algo atrás). curl: dedos (0 reposo, 1 puño, <0 abiertos).
        wrist: flexión de muñeca (grados, + hacia la palma). shrug/protract: clavícula extra (además de la automática)."""
        c = self.chest_q
        d0, p0, h0 = ARM_REST[side]
        u = (c @ side_vec(upper, side)).normalized()
        l = (c @ side_vec(lower, side)).normalized()
        pd = (c @ side_vec(palm if palm is not None else D(-0.25, -1, 0), side)).normalized()
        # Clavícula automática (ritmo escápulo-humeral): al subir el brazo sube el hombro; al adelantarlo, se adelanta.
        uc = c.inverted() @ u
        elev = math.degrees(math.acos(max(-1.0, min(1.0, -uc.z))))           # 0 = colgando, 180 = arriba del todo
        reach = uc.dot(FWD)
        up_deg = max(0.0, elev - 65.0) / 115.0 * 26.0 + shrug
        fw_deg = max(0.0, reach) * 14.0 - max(0.0, -reach) * 8.0 + protract
        sg = 1 if side == 'R' else -1
        self.rot[f'Shoulder.{side}'] = c @ q(Vector((0, sg, 0)), up_deg) @ q(Vector((0, 0, sg)), fw_deg)
        Rsh = self.rot[f'Shoulder.{side}']
        swing = (Rsh @ BONE_DIR[f'UpperArm.{side}']).rotation_difference(u) @ Rsh
        # Bisagra preferida: la que lleva el brazo sin girarlo sobre su eje (giro del húmero 0). Con el brazo casi recto
        # se usa ésta y la palma la orienta el antebrazo; doblado, manda el plano brazo-antebrazo.
        n_pref = swing @ h0
        n_pref = (n_pref - u * u.dot(n_pref)).normalized()
        n = u.cross(l)
        bend = math.degrees(u.angle(l)) if u.length and l.length else 0.0
        if bend < 8.0 or n.length < 1e-4:
            n = n_pref
        else:
            n = n.normalized()
            if n.dot(n_pref) < -0.34:
                WARN.setdefault(CUR_CLIP[0], set()).add(side)
                if os.environ.get('RIGDBG'): print(f'  REVÉS {CUR_CLIP[0]} {side} t={CUR_CLIP[1]:.2f} codo {bend:.0f} dot {n.dot(n_pref):.2f}')
            n = n.lerp(n_pref, max(0.0, (22.0 - bend) / 22.0) * 0.6).normalized()
        Ru = frame_rot(BONE_DIR[f'UpperArm.{side}'], h0, u, n)
        Rl = frame_rot(BONE_DIR[f'LowerArm.{side}'], h0, l, n)
        # Mano: dirección del antebrazo flexionada en la muñeca hacia la palma; el giro se reparte con el antebrazo.
        palm_l = (Rl @ p0)
        pd_o = (pd - l * l.dot(pd))
        if pd_o.length < 1e-4: pd_o = palm_l
        pd_o.normalize()
        tw = math.degrees(palm_l.angle(pd_o)) if palm_l.length else 0.0
        if l.dot(palm_l.cross(pd_o)) < 0: tw = -tw
        tw = max(-100.0, min(100.0, tw))                 # pronación/supinación del antebrazo (rango humano)
        Rtw = q(l, tw * 0.55) @ Rl
        Rh0 = q(l, tw) @ Rl
        hw0 = d0.cross(p0).normalized()                 # eje de flexión de muñeca y dedos: hacia la palma
        hinge_h = Rh0 @ hw0
        wrist = max(-55.0, min(65.0, wrist))             # muñeca: flexión hacia la palma +, extensión -
        Rh = q(hinge_h, wrist) @ Rh0
        # Ayudantes de volumen: codo a mitad de camino entre brazo y antebrazo; deltoides a mitad entre la clavícula
        # y el brazo (sólo la elevación, sin el giro del brazo, para que el hombro no se retuerza).
        self.rot[f'Deltoid.{side}'] = qslerp(Rsh, swing, 0.5)
        self.rot[f'Elbow.{side}'] = qslerp(Ru, Rl, 0.5)
        hum = math.degrees((swing.inverted() @ Ru).angle)
        st = TWIST_STATS.setdefault(CUR_CLIP[0], [0.0, 0.0, 0.0])
        st[0] = max(st[0], abs(tw)); st[1] = max(st[1], hum if hum <= 180 else 360 - hum); st[2] = max(st[2], bend)
        self.rot[f'UpperArm.{side}'] = Ru
        self.rot[f'LowerArm.{side}'] = Rl
        self.rot[f'LowerArmTwist.{side}'] = Rtw
        self.rot[f'Hand.{side}'] = Rh
        # Dedos (bloque de la manopla) y pulgar.
        hh = Rh @ hw0
        self.rot[f'Fingers1.{side}'] = q(hh, 32.0 * curl) @ Rh
        self.rot[f'Fingers2.{side}'] = q(hh, 70.0 * curl) @ Rh
        if f'Thumb1.{side}' in J:
            tcurl = curl if thumb is None else thumb
            hd = Rh @ d0
            self.rot[f'Thumb1.{side}'] = q(hd, sg * 18.0 * tcurl) @ q(hh, 10.0 * tcurl) @ Rh
            self.rot[f'Thumb2.{side}'] = q(hd, sg * 30.0 * tcurl) @ q(hh, 20.0 * tcurl) @ Rh
        return self

    def leg(self, side, swing=0.0, knee=0.0, foot=0.0, spread=0.0, toe=0.0):
        """swing + = pierna adelante; knee = flexión; foot + = punta arriba; toe + = dedos arriba (despegue)."""
        h = self.hips_q
        rs = q(ROLL, spread if side == 'R' else -spread)
        up_dir = h @ (rs @ q(PITCH, -swing) @ Vector((0, 0, -1)))
        lo_dir = h @ (rs @ q(PITCH, -swing + knee) @ Vector((0, 0, -1)))
        fh, ft = J[f'Foot.{side}']
        f_dir = h @ (q(PITCH, -swing + knee - foot) @ (ft - fh).normalized())
        th, tt = J[f'Toe.{side}']
        t_dir = h @ (q(PITCH, -swing + knee - foot - toe) @ (tt - th).normalized())
        self.rot[f'UpperLeg.{side}'] = aim(f'UpperLeg.{side}', up_dir)
        self.rot[f'LowerLeg.{side}'] = aim(f'LowerLeg.{side}', lo_dir)
        self.rot[f'Foot.{side}'] = aim(f'Foot.{side}', f_dir)
        self.rot[f'Toe.{side}'] = aim(f'Toe.{side}', t_dir)
        return self


LEG = J['UpperLeg.L'][0].z - J['Foot.L'][0].z


def drop_for(knee_deg, swing_deg=0.0):
    k, s = math.radians(knee_deg), math.radians(swing_deg)
    half = LEG / 2
    return half * (1 - math.cos(s)) + half * (1 - math.cos(k - s))


# ---------------- utilidades de animación (curvas suaves, retrasos, rebotes)
TAU = 2 * math.pi


def S(t, a=1.0):
    return math.sin(TAU * t) * a


def C(t, a=1.0):
    return math.cos(TAU * t) * a


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def ease_out(u):
    u = max(0.0, min(1.0, u))
    return 1 - (1 - u) ** 3


def back(u, s=1.6):
    """Llega pasándose un poco y vuelve (overshoot) — para asentar poses con gracia."""
    u = max(0.0, min(1.0, u)) - 1
    return u * u * ((s + 1) * u + s) + 1


def K(t, *pts):
    """Interpolación suave entre (tiempo, valor)."""
    if t <= pts[0][0]: return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0))
            if isinstance(v0, tuple): return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return pts[-1][1]


def mix(a, b, k):
    return (a * (1 - k) + b * k).normalized()


def fwd(deg):
    return q(PITCH, -deg)


# Separación del brazo colgando: la cara interior de la manga, a la altura del codo, no debe meterse en el torso.
_LSE = float(np.linalg.norm(ARM['E'] - ARM['S']))
_ze = float(ARM['S'][2]) - _LSE
_band = (np.abs(CO[:, 2] - _ze) < 0.012) & ~ARM_MASK & ~head_only      # torso a esa altura (sin mangas ni pelo)
_tw = float(np.abs(CO[_band, 0]).max()) if _band.any() else float(ARM['S'][0])
_abd = math.degrees(math.asin(max(-1.0, min(1.0, (_tw + 0.8 * ARM['r_el'] - float(ARM['S'][0])) / _LSE))))
ABD = max(10.0, min(38.0, _abd))
AB = math.tan(math.radians(ABD))
print(f'BRAZO colgando: torso a la altura del codo {_tw:.3f}, separación {ABD:.0f}° (pedía {_abd:.0f}°)')
ARM_DOWN = D(0, AB, -1)        # brazo relajado (separado lo justo para no meterse en el cuerpo chibi)
PALM_IN = D(-0.25, -1, 0)      # palma hacia el muslo
PALM_BACK = D(-1, -0.3, 0)
PALM_DOWN = D(0, 0, -1)
PALM_UP = D(0, -0.2, 1)
PALM_FWD = D(1, 0, 0)


def relaxed_arm(p, side, swing=0.0, bend=14.0, curl=0.35, spread=0.0, palm=None):
    u = fwd(swing) @ D(0, AB + spread, -1)
    l = fwd(swing + bend) @ D(0, AB - 0.04 + spread, -1)
    return p.arm(side, u, l, palm=palm or PALM_IN, curl=curl, wrist=4 + bend * 0.15)


CLIPS = {}
CUR_CLIP = ['-', 0.0]


def clip(sec, loop=False):
    def deco(fn):
        CLIPS[fn.__name__] = (sec, loop, fn)
        return fn
    return deco


# ================================================================ clips de cuerpo entero
@clip(3.2, True)
def Idle(t):
    """Respira (pecho y hombros), pasa el peso de un pie a otro y mira alrededor de vez en cuando."""
    br = S(t * 2)                                   # 2 respiraciones por ciclo
    shift = S(t)                                    # peso a un lado y a otro
    look = K(t, (0, 0), (0.28, 0), (0.36, 22), (0.5, 22), (0.58, 0), (0.75, 0), (0.82, -14), (0.9, -14), (1, 0))
    nod = K(t, (0, 0), (0.36, -4), (0.5, -2), (0.58, 0), (0.82, 3), (1, 0))
    p = Pose().torso(hips=(1, shift * 2, shift * 2.5), spine=(2 + br * 0.6, -shift, -shift * 1.5),
                     chest=(br * 1.8, look * 0.15, -shift), neck=(0, look * 0.3, 0), head=(-3 + nod, look * 0.7, shift * 2 + S(t * 3, 0.8)))
    p.hips = Vector((shift * 0.012 * H, 0, S(t * 2, 0.003 * H)))
    p.leg('L', 0, 4 + max(0, -shift) * 8, spread=1).leg('R', 0, 4 + max(0, shift) * 8, spread=1)
    lag = 0.06
    for side, o in (('L', 0.0), ('R', 0.5)):
        sh = S(t * 2 - lag, 1.5)
        relaxed_arm(p, side, S(t + o - lag, 2.5), 14 + S(t + o, 3), curl=0.32 + S(t * 2 + o, 0.05))
        p.rot[f'Shoulder.{side}'] = p.rot[f'Shoulder.{side}'] @ q(Vector((0, 1 if side == 'R' else -1, 0)), 1.5 + sh)
    return p


# ---------------- locomoción con apoyo PLANTADO (QA 0.9.3: "parece que patinan")
# Antes Walk/Run eran de péndulo (la pierna oscilaba por ángulos y el pie nunca quedaba quieto en el suelo): con ningún
# ritmo dejaban de patinar. Ahora cada pie sigue una trayectoria en el espacio del clip (sin avance): en el APOYO la bola
# del pie (cabeza del hueso Toe, la que mide el juego) va hacia atrás en línea recta a velocidad constante = velocidad de
# diseño del clip, plana y luego girando sobre la bola (despega el talón); en el BALANCEO vuelve por delante con una curva
# de Hermite cuya velocidad al principio y al final es la del apoyo (sin tirón al tocar suelo). Piernas por IK de dos
# huesos (rodilla hacia delante); la cadera baja lo justo para que la pierna llegue (constante por clip) más el rebote.
# Fase: el pie izquierdo toca suelo en t = 0.25 (cuando el torso de siempre lo tiene delante: sw > 0), el derecho en 0.75.
_TH = {s_: (J[f'UpperLeg.{s_}'][1] - J[f'UpperLeg.{s_}'][0]).length for s_ in 'LR'}
_SH = {s_: (J[f'LowerLeg.{s_}'][1] - J[f'LowerLeg.{s_}'][0]).length for s_ in 'LR'}
_VF = {s_: J[f'Foot.{s_}'][1] - J[f'Foot.{s_}'][0] for s_ in 'LR'}           # tobillo → bola, en reposo
_VT = {s_: (J[f'Toe.{s_}'][1] - J[f'Toe.{s_}'][0]).normalized() for s_ in 'LR'}
_BALL = {s_: J[f'Toe.{s_}'][0].copy() for s_ in 'LR'}                           # bola del pie en reposo
_HIPS_HEAD = J['Hips'][0].copy()
LOCO = {}           # medidas por clip (para el log)


def _hermite(u, p0, p1, m0, m1):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def foot_track(ph, P):
    """Bola del pie en el espacio del clip para la fase ph (0 = toca suelo). Devuelve (adelante, altura, inclinación del pie
    en grados (+ = punta arriba), flexión de los dedos (+ = dedos arriba respecto al pie))."""
    ph %= 1.0
    b, Ls, c = P['beta'], P['Ls'], P['c']
    if ph < b:                                       # apoyo: hacia atrás a velocidad constante
        u = ph / b
        f = c + Ls / 2 - Ls * u
        k = max(0.0, (u - P['roll']) / (1 - P['roll']))
        pitch = -P['off'] * k * k                    # el talón se despega girando sobre la bola
        return f, 0.0, pitch, -pitch
    u = (ph - b) / (1 - b)                           # balanceo: vuelve por delante
    m = -Ls / b * (1 - b)                            # misma velocidad que el apoyo (en unidades de u)
    f = _hermite(u, c - Ls / 2, c + Ls / 2, m, m)
    up = P['peak']
    if u < up: h = P['lift'] * math.sin(0.5 * math.pi * u / up)
    else: h = P['lift'] * math.cos(0.5 * math.pi * (u - up) / (1 - up))
    pitch = K(u, (0, -P['off']), (0.3, P['swing_pitch'] * 0.6), (0.65, P['swing_pitch']), (1, 0))
    toe = K(u, (0, P['off']), (0.25, 0), (1, 0))
    return f, h, pitch, toe


def _hip_joint(p, side):
    return _HIPS_HEAD + p.hips + p.hips_q @ (J[f'UpperLeg.{side}'][0] - _HIPS_HEAD)


def _ankle_target(side, ph, P):
    f, h, pitch, toe = foot_track(ph, P)
    sgn = 1 if side == 'L' else -1
    ball = Vector((sgn * abs(_BALL[side].x) * P['narrow'], _BALL[side].y - f, _BALL[side].z + h))
    return ball - q(PITCH, -pitch) @ _VF[side], ball, pitch, toe


def leg_ik(p, side, ankle, pitch, toe):
    """Pierna por IK: muslo/espinilla hasta el tobillo, rodilla hacia delante (con el giro de la cadera), pie y dedos
    orientados en el espacio del clip (planos en el suelo, sin girar con la cadera)."""
    hj = _hip_joint(p, side)
    a, b_ = _TH[side], _SH[side]
    d = ankle - hj
    dist = min(d.length, (a + b_) * 0.999)
    dn = d.normalized()
    ca = max(-1.0, min(1.0, (a * a + dist * dist - b_ * b_) / (2 * a * dist)))
    kd = p.hips_q @ FWD
    kd = (kd - dn * kd.dot(dn)).normalized()
    knee = hj + a * (ca * dn + math.sqrt(max(0.0, 1 - ca * ca)) * kd)
    p.rot[f'UpperLeg.{side}'] = aim(f'UpperLeg.{side}', (knee - hj).normalized())
    p.rot[f'LowerLeg.{side}'] = aim(f'LowerLeg.{side}', (hj + dn * dist - knee).normalized())
    p.rot[f'Foot.{side}'] = aim(f'Foot.{side}', (q(PITCH, -pitch) @ _VF[side]).normalized())
    p.rot[f'Toe.{side}'] = aim(f'Toe.{side}', q(PITCH, -(pitch + toe)) @ _VT[side])
    return p


def plant_legs(p, t, P):
    """Coloca las dos piernas con el apoyo plantado y fija la altura de cadera: rebote del clip + bajada constante (la
    mínima para que la pierna llegue a la bola del pie en todo el ciclo, calculada una vez por clip)."""
    name = P['name']
    if name not in LOCO:
        # Centro del apoyo (c) que menos obliga a bajar la cadera, y esa bajada (sólo cuentan los pies apoyados).
        best = None
        for c_ in np.linspace(-0.08, 0.06, 29) * _LK:
            P['c'] = float(c_)
            nd = _need_drop(P)
            if best is None or nd < best[1] - 1e-5: best = (float(c_), nd)
        P['c'] = best[0]
        LOCO[name] = dict(drop=best[1] + 0.002, v=P['Ls'] / (P['beta'] * P['sec']), c=best[0])
        print(f'LOCO {name}: paso de apoyo {P["Ls"]:.3f}, apoyo {P["beta"]:.2f} del ciclo, centro {best[0]:+.3f}, '
              f'velocidad de diseño {LOCO[name]["v"]:.3f} u/s (en el clip), cadera baja {LOCO[name]["drop"]:.3f}')
    p.hips = Vector((0, 0, P['bob'](t) - LOCO[name]['drop']))
    for side, o in (('L', -0.25), ('R', 0.25)):
        ank, ball, pitch, toe = _ankle_target(side, t + o, P)
        leg_ik(p, side, ank, pitch, toe)
    return p


def _need_drop(P):
    """Bajada de cadera mínima para que la pierna (al reach de su largo) llegue a la bola del pie en todo el apoyo."""
    need = 0.0
    for i in range(60):
        tt = i / 60
        pp = P['torso'](tt)
        pp.hips = Vector((0, 0, P['bob'](tt)))
        for side, o in (('L', -0.25), ('R', 0.25)):
            if (tt + o) % 1.0 >= P['beta']: continue          # sólo el apoyo tiene que llegar al suelo
            ank = _ankle_target(side, tt + o, P)[0]
            hj = _hip_joint(pp, side)
            L = (_TH[side] + _SH[side]) * P['reach']
            dh = math.hypot(ank.x - hj.x, ank.y - hj.y)
            if dh >= L:
                need = max(need, hj.z - ank.z)
                continue
            need = max(need, hj.z - (ank.z + math.sqrt(L * L - dh * dh)))
    return need


def _walk_torso(t):
    sw = 26 * math.sin(TAU * t)
    bob = C(2 * (t - 0.31))
    return Pose().torso(hips=(5, sw * 0.3, -S(t, 3)), spine=(2 + bob * 1.2, -sw * 0.25, S(t, 1.5)),
                        chest=(-1, -sw * 0.35, S(t - 0.05, 1.5)), neck=(0, sw * 0.15, 0), head=(-5 + bob * 1.5, sw * 0.2, -S(t, 2)))


_LK = LEG / 0.255          # proporción de pierna respecto a la de referencia (Lola: 0.255)
WALK = dict(name='Walk', sec=0.6, beta=0.56, Ls=0.20 * _LK, c=0.0, roll=0.55, off=28.0, lift=0.045 * _LK,
            peak=0.45, swing_pitch=8.0, narrow=0.92, reach=0.975, torso=_walk_torso,
            bob=lambda t: -0.012 * H * C(2 * (t - 0.31)))        # abajo justo tras el apoyo (0.25 / 0.75), arriba al pasar


@clip(WALK['sec'], True)
def Walk(t):
    """Paso chibi con rebote y el pie de apoyo PLANTADO: cadera que gira con la pierna, hombros al contrario, brazos con
    retraso y cabeza estable."""
    p = _walk_torso(t)
    plant_legs(p, t, WALK)
    lag = 0.07                                       # los brazos siguen al cuerpo con retraso (superposición)
    for side, sgn in (('L', -1), ('R', 1)):
        a = sgn * 26 * math.sin(TAU * (t - lag))
        relaxed_arm(p, side, a, 16 + max(0.0, a) * 0.9, curl=0.4)
    return p


_RUN_MID = 0.25 + 0.36 / 2          # mitad del apoyo izquierdo
RUN_OFF = 0.25 - (1 - 2 * 0.36) / 4   # mitad del vuelo que acaba con el apoyo izquierdo


def _run_torso(t):
    sw = 42 * math.sin(TAU * t)
    fl = 0.5 - 0.5 * C(2 * (t - _RUN_MID))           # 0 a mitad del apoyo, 1 en el vuelo
    return Pose().torso(hips=(14, sw * 0.32, -S(t, 4)), spine=(6 + fl * 2, -sw * 0.3, 0),
                        chest=(3, -sw * 0.45, S(t - 0.05, 2)), neck=(0, sw * 0.15, 0), head=(-14 - fl * 2, sw * 0.25, 0))


RUN = dict(name='Run', sec=0.5, beta=0.36, Ls=0.26 * _LK, c=0.0, roll=0.4, off=38.0, lift=0.10 * _LK,
           peak=0.38, swing_pitch=12.0, narrow=0.85, reach=0.985, torso=_run_torso,
           bob=lambda t: -0.03 * H * C(2 * (t - _RUN_MID)))      # abajo a mitad del apoyo, arriba en el vuelo


@clip(RUN['sec'], True)
def Run(t):
    """Carrera con fase de vuelo y pie de apoyo PLANTADO: inclinado, brazos bombeando con los codos a 90° y puños cerrados.
    El clip empieza a mitad de un vuelo (t + RUN_OFF): la costura del bucle no cae con un pie apoyado."""
    t = (t + RUN_OFF) % 1.0
    p = _run_torso(t)
    plant_legs(p, t, RUN)
    lag = 0.05
    for side, sgn in (('L', -1), ('R', 1)):
        a = sgn * 46 * math.sin(TAU * (t - lag))
        u = fwd(a + 8) @ D(0, AB * 0.9, -1)
        l = fwd(a + 98) @ D(0, -0.05, -1)
        p.arm(side, u, l, palm=PALM_IN, curl=0.85, wrist=8)
    return p


@clip(0.4)
def Jump(t):
    """Anticipación (se agacha, brazos atrás) y despegue explosivo (brazos arriba, puntas de los pies)."""
    k = K(t, (0, 0), (0.42, 1), (0.62, 0), (1, 0))   # agacharse
    up = ease_out((t - 0.42) / 0.58) if t > 0.42 else 0.0
    p = Pose().torso(hips=(18 * k - 4 * up, 0, 0), spine=(10 * k - 6 * up, 0, 0), chest=(4 * k - 4 * up, 0, 0), head=(-14 * k + 6 * up, 0, 0))
    p.leg('L', 18 * k + 4 * up, 48 * k + 6 * up, foot=-12 * k - 25 * up, toe=-10 * up).leg('R', 18 * k + 4 * up, 48 * k + 6 * up, foot=-12 * k - 25 * up, toe=-10 * up)
    p.hips = Vector((0, 0, -drop_for(48 * k, 18 * k)))
    for side in 'LR':
        u = mix(mix(ARM_DOWN, D(-0.7, 0.3, -0.6), k), D(0.55, 0.25, 0.8), up)
        l = mix(mix(fwd(14) @ ARM_DOWN, D(-0.5, 0.2, -0.7), k), D(0.45, 0.1, 0.9), up)
        p.arm(side, u, l, palm=PALM_IN, curl=0.5 - 0.6 * up)
    return p


@clip(0.8, True)
def Air(t):
    """En el aire: brazos abiertos para equilibrar (con un leve aleteo), piernas recogidas."""
    p = Pose().torso(hips=(-4, 0, S(t, 2)), spine=(-4, 0, 0), chest=(-2, S(t, 3), 0), head=(4 + S(t - 0.1, 2), S(t - 0.15, 3), 0))
    p.leg('L', 30 + S(t, 5), 60, foot=-18, toe=-5).leg('R', 8 - S(t, 5), 38, foot=-18, toe=-5)
    for side, o in (('L', 0), ('R', 0.5)):
        fl = S(t + o, 10)
        p.arm(side, D(0.2 + fl * 0.004, 0.9, 0.3 + fl * 0.01), D(0.6 + fl * 0.01, 0.75, 0.3 + fl * 0.015), palm=PALM_DOWN, curl=-0.2)
    return p


@clip(0.42)
def Land(t):
    """Impacto: se aplasta (rodillas y espalda), los brazos siguen bajando por inercia y se asienta con rebote."""
    k = K(t, (0, 0.55), (0.18, 1), (0.55, 0.15), (0.75, 0.22), (1, 0))
    arms = K(t, (0, 0.4), (0.25, 1), (1, 0))
    p = Pose().torso(hips=(16 * k, 0, 0), spine=(12 * k, 0, 0), chest=(6 * k, 0, 0), head=(-16 * k + 4 * arms, 0, 0))
    p.leg('L', 22 * k, 60 * k, foot=-12 * k).leg('R', 22 * k, 60 * k, foot=-12 * k)
    p.hips = Vector((0, 0, -drop_for(60 * k, 22 * k)))
    for side in 'LR':
        u = mix(ARM_DOWN, D(0.35, 0.8, -0.5), arms)
        l = mix(fwd(14) @ ARM_DOWN, D(0.55, 0.65, -0.3), arms)
        p.arm(side, u, l, palm=PALM_DOWN, curl=0.1 - 0.3 * arms)
    return p


@clip(0.9)
def Slip(t):
    """Resbalón: los pies se van hacia delante, los brazos saltan arriba con retraso y la cabeza da un latigazo."""
    k = K(t, (0, 0), (0.32, 1), (1, 1))
    ka = K(t, (0, 0), (0.12, 0), (0.42, 1), (1, 0.85))           # brazos con retraso
    whip = K(t, (0, 0), (0.3, -15), (0.5, 30), (0.7, 18), (1, 22))   # cabeza: atrás y adelante
    flail = math.sin(t * 34) * (1 - k * 0.5)
    p = Pose().torso(hips=(-52 * k, 0, 6 * flail), spine=(-12 * k, 0, 0), chest=(6 * k, 0, 0), head=(whip, 0, 10 * flail))
    p.leg('L', 75 * k, 22 * k, foot=22 * k).leg('R', 52 * k, 42 * k, foot=22 * k)
    p.hips = Vector((0, 0, -LEG * 0.78 * K(t, (0, 0), (0.55, 1), (0.7, 0.95), (1, 1))))
    for side, o in (('L', 0), ('R', 1.3)):
        f = 25 * math.sin(t * 26 + o) * (1 - ka * 0.5)
        fa = f / 25.0
        p.arm(side, mix(ARM_DOWN, D(0.25 + 0.15 * fa, 0.85, 0.65 + 0.2 * fa), ka), mix(fwd(14) @ ARM_DOWN, D(0.65 + 0.2 * fa, 0.55, 0.75 + 0.2 * fa), ka),
              palm=mix(PALM_IN, PALM_FWD, ka), curl=-0.4 * ka)
    return p


@clip(1.6, True)
def Lie(t):
    """En el suelo: respira, la cabeza se ladea, los dedos tiemblan."""
    br = S(t)
    p = Pose().torso(hips=(-52, 0, 0), spine=(-12 + br, 0, 0), chest=(6 + br * 1.5, 0, 0), head=(24 + S(t, 3), S(t * 0.5, 12), S(t * 0.5 + 0.2, 6)))
    p.leg('L', 75, 22, foot=22).leg('R', 52, 42, foot=22)
    p.hips = Vector((0, 0, -LEG * 0.78))
    for side, o in (('L', 0), ('R', 0.5)):
        p.arm(side, fwd(60) @ D(0, 1, -0.3), fwd(62) @ D(0, 1, -0.2), palm=PALM_DOWN, curl=0.15 + S(t * 2 + o, 0.08))
    return p


@clip(0.85)
def GetUp(t):
    """Se incorpora: se sienta, apoya una mano, empuja, se pone de pie y se sacude."""
    k = 1 - K(t, (0, 0), (0.8, 1))
    push = math.sin(math.pi * min(1.0, t / 0.75))
    shake = math.sin(t * 40) * max(0.0, (t - 0.78) / 0.22) * 4
    p = Pose().torso(hips=(-52 * k + 22 * push, 0, 0), spine=(-12 * k + 22 * push, 0, shake), chest=(6 * k + 6 * push, 0, 0),
                     head=(24 * k - 12 * push, shake * 2, 0))
    p.leg('L', 75 * k + 32 * push, 22 * k + 75 * push, foot=22 * k).leg('R', 52 * k, 42 * k + 32 * push, foot=22 * k)
    p.hips = Vector((0, 0, -LEG * 0.78 * k))
    p.arm('R', mix(D(0.2, 1, -0.4), D(-0.3, 0.5, -1), push), mix(D(0.2, 1, -0.3), D(-0.2, 0.4, -1), push), palm=PALM_DOWN, curl=-0.3 * push, wrist=-30 * push)
    relaxed_arm(p, 'L', 20 * push, 30 * push + 14, curl=0.3)
    return p


@clip(1.1, True)
def Stunned(t):
    """Mareado: tambaleo en círculos, la cabeza dibuja un ocho con retraso, brazos sueltos que se balancean."""
    p = Pose().torso(hips=(S(t, 4), 0, S(t + 0.25, 6)), spine=(5, 0, S(t + 0.2, 5)), chest=(2, S(t + 0.3, 4), S(t + 0.3, 3)),
                     neck=(0, S(t + 0.35, 6), 0), head=(12 + S(2 * t + 0.4, 6), S(t + 0.4, 16), S(2 * t + 0.4, 10)))
    p.leg('L', S(t, 6), 14).leg('R', -S(t, 6), 14)
    p.hips = Vector((S(t + 0.25, 0.01 * H), 0, -drop_for(14)))
    for side, o in (('L', 0.1), ('R', 0.6)):
        relaxed_arm(p, side, S(t + o, 12), 18 + S(t + o, 8), curl=0.1, spread=0.25 + S(t + o, 0.1))
    return p


@clip(1.0, True)
def Victory(t):
    """¡Ganamos!: saltitos, puños al aire alternando, la cabeza atrás de alegría."""
    b = abs(math.sin(TAU * t))
    al = S(t)
    p = Pose().torso(hips=(-4, al * 6, 0), spine=(-6, al * 4, 0), chest=(-4 - b * 3, al * 4, al * 3), head=(-18 - b * 6, S(t - 0.1, 10), al * 4))
    p.leg('L', 0, 10 + b * 28, foot=-8 * (1 - b)).leg('R', 0, 10 + b * 28, foot=-8 * (1 - b))
    p.hips = Vector((0, 0, -drop_for(10 + b * 28) + (1 - b) * 0.07 * H))
    for side, o in (('L', 0.0), ('R', 0.5)):
        pump = 0.5 + 0.5 * math.sin(TAU * (t + o))
        p.arm(side, D(0.15, 0.8, 1), mix(D(0.35, 0.45, 0.75), D(0.1, 0.5, 1), pump), palm=PALM_FWD, curl=1.0, shrug=4 * pump)
    return p


@clip(1.6, True)
def Defeat(t):
    """Derrota: hombros caídos hacia delante, cabeza gacha que niega y un suspiro."""
    sigh = K(t, (0, 0), (0.35, 1), (0.6, 0.2), (1, 0))
    p = Pose().torso(hips=(6, 0, 0), spine=(14 + sigh * 3, 0, 0), chest=(10 - sigh * 4, 0, 0), neck=(6, 0, 0), head=(24 - sigh * 6, S(t, 14), 0))
    p.leg('L', 0, 6, spread=1).leg('R', 0, 6, spread=1)
    p.hips = Vector((0, 0, -drop_for(6)))
    for side in 'LR':
        u = fwd(6) @ D(0, AB * 0.75, -1)
        l = fwd(14) @ D(0, AB * 0.65, -1)
        p.arm(side, u, l, palm=PALM_BACK, curl=0.15, protract=10, shrug=-6 + sigh * 8)
    return p


# ================================================================ gestos (rueda de gestos del juego)
def flapf(x):
    """Aleteo de un tiempo: sube rápido pasándose un poco y baja suave."""
    x %= 1.0
    return back(x * 2.2) if x < 0.45 else 1 - ease((x - 0.45) / 0.55)


@clip(1.6, True)
def Dance(t):
    """El baile del pollito: codos que aletean como alas, rodillas al compás, cadera de lado a lado y la cabeza
    que picotea adelante y atrás como una gallina. Dos tiempos por ciclo, cada aleteo con rebote."""
    beat = (t * 4) % 1.0
    bounce = abs(math.sin(TAU * t * 2))                                            # 4 flexiones por ciclo
    sway = S(t)
    peck = K(beat, (0, 0), (0.18, 1), (0.5, -0.4), (1, 0))
    p = Pose().torso(hips=(2, sway * 10, sway * 6), spine=(2 + bounce * 2, -sway * 5, -sway * 3),
                     chest=(-2 + peck * 4, -sway * 6, sway * 2), neck=(-peck * 10, 0, 0),
                     head=(-peck * 14 + 4, S(t - 0.08, 8), sway * 6))
    knee = 10 + bounce * 16
    p.leg('L', 0, knee + max(0, sway) * 8, foot=-2, spread=4).leg('R', 0, knee + max(0, -sway) * 8, foot=-2, spread=4)
    p.hips = Vector((sway * 0.025 * H, 0, -drop_for(knee) + bounce * 0.012 * H))
    for side, o in (('L', 0.0), ('R', 0.03)):          # el derecho, un pelín tarde: no parece un robot
        f = flapf(t * 4 + o)
        u = D(-0.3, 1.0, -0.5 + 0.55 * f)             # el codo sale hacia fuera (y algo atrás) y sube con el aleteo
        l = D(0.6, -0.75, -0.42)             # la mano vuelve delante del sobaco, por debajo del pecho (ala plegada)
        p.arm(side, u, l, palm=PALM_DOWN, curl=0.9, wrist=10, shrug=6 * f)
    return p


@clip(1.2, True)
def Wave(t):
    """¡Hola!: brazo derecho arriba saludando con la mano abierta (antebrazo de lado a lado con retraso en la mano),
    cabeza ladeada con una sonrisa y el peso en la pierna contraria."""
    w = S(t * 2)                                       # dos vaivenes por ciclo
    wl = S(t * 2 - 0.07)                               # la mano llega un poco después que el antebrazo
    p = Pose().torso(hips=(1, -4, -4), spine=(1, 4, 5), chest=(-3, 6, 6 + w * 1.5), neck=(0, 4, 0),
                     head=(-5, 8, -14 + S(t * 2 - 0.15, 3)))
    p.leg('L', 0, 5, spread=2).leg('R', 2, 12, foot=-4, spread=3)
    p.hips = Vector((-0.012 * H, 0, -drop_for(8)))
    # Codo fuera a la altura del hombro (brazos chibi cortos: así la mano queda por encima y lejos del sombrero).
    p.arm('R', D(0.1, 1.0, 0.2), D(0.05, 0.35 + 0.4 * w, 1.0), palm=PALM_FWD, curl=-0.35, wrist=-12 * wl, shrug=10)
    relaxed_arm(p, 'L', 4 + S(t - 0.1, 3), 16, curl=0.35)
    return p


@clip(1.4, True)
def Taunt(t):
    """Burla: manos en jarras, se echa atrás riéndose a carcajadas (hombros que se sacuden) y menea la cadera."""
    laugh = abs(math.sin(TAU * t * 5)) * K(t, (0, 0.4), (0.3, 1), (0.8, 1), (1, 0.4))
    wig = S(t * 2)
    p = Pose().torso(hips=(-2, wig * 8, wig * 4), spine=(-8 - laugh * 3, -wig * 3, 0), chest=(-10 - laugh * 5, -wig * 4, 0),
                     neck=(-6, 0, 0), head=(-14 - laugh * 8, S(t * 2, 12), S(t - 0.1, 4)))
    p.leg('L', 0, 8 + max(0, wig) * 10, spread=6).leg('R', 0, 8 + max(0, -wig) * 10, spread=6)
    p.hips = Vector((wig * 0.02 * H, 0, -drop_for(9) + laugh * 0.006 * H))
    for side in 'LR':
        # Puño apoyado en la cadera: codo hacia fuera y algo atrás, antebrazo que vuelve hacia la cintura.
        p.arm(side, D(-0.25, 1.0, -0.75), D(0.35, -0.85, -0.35), palm=PALM_DOWN, curl=0.85, wrist=15, shrug=2 + laugh * 5)
    return p


# ================================================================ capa de brazos (pecho, brazos y cabeza)
def upper():
    p = Pose().torso()
    p.leg('L').leg('R')
    return p


@clip(1.0, True)
def Carry(t):
    """Sujeta a la gallina contra el pecho con las dos manos ahuecadas; se revuelve y los brazos la compensan."""
    st = S(t * 2)
    p = upper().torso(spine=(-4, 0, 0), chest=(-4, st * 3, st * 1.5), head=(10, -st * 3, 0))
    for side, o in (('L', 0.0), ('R', 0.5)):
        j = S(t * 2 + o - 0.08, 0.06)
        p.arm(side, D(0.55, 0.45, -0.7), D(0.7, -0.7, 0.15 + j), palm=PALM_UP, curl=0.65, wrist=12, protract=6)
    return p


@clip(0.6)
def Throw(t):
    """Anticipación (se echa atrás, brazo por detrás de la cabeza), latigazo adelante, suelta y acompaña."""
    k1 = K(t, (0, 0), (0.38, 1), (0.52, 0))
    k2 = K(t, (0.38, 0), (0.55, 1), (0.8, 0.6), (1, 0))
    rel = 1.0 if 0.5 < t < 0.85 else 0.0
    p = upper().torso(spine=(-14 * k1 + 22 * k2, 25 * k1 - 32 * k2, 0), chest=(-10 * k1 + 16 * k2, 12 * k1 - 18 * k2, 0),
                      head=(6 * k1 - 8 * k2, -8 * k1 + 10 * k2, 0))
    for side in 'LR':
        u = (D(0.55, 0.45, -0.7) * (1 - k1 - k2) + D(-0.4, 0.45, 0.8) * k1 + D(1, 0.15, 0.1) * k2).normalized()
        l = (D(0.7, -0.7, 0.15) * (1 - k1 - k2) + D(-0.2, -0.3, 1) * k1 + D(1, 0.0, -0.15) * k2).normalized()
        pal = (PALM_UP * max(0.0, 1 - k1 - k2) + PALM_FWD * k1 + PALM_DOWN * k2).normalized()
        p.arm(side, u, l, palm=pal, curl=0.65 * (1 - rel) - 0.4 * rel, wrist=-20 * k1 + 25 * k2)
    return p


@clip(0.6)
def NetSwing(t):
    """Red: carga hacia atrás y arriba, barrido en arco hasta el otro lado y acompaña (la izquierda equilibra)."""
    k1 = K(t, (0, 0), (0.32, 1), (0.48, 0))
    k2 = K(t, (0.32, 0), (0.52, 1), (0.75, 0.75), (1, 0))
    p = upper().torso(spine=(6 * k2, 12 * k1 - 18 * k2, 0), chest=(6 * k2 - 3 * k1, 32 * k1 - 48 * k2, 0), head=(-4 * k2, -10 * k1 + 15 * k2, 0))
    rest_u, rest_l = ARM_DOWN, fwd(14) @ ARM_DOWN
    u = (rest_u * (1 - k1 - k2) + D(-0.35, 0.75, 0.6) * k1 + D(1, -0.45, 0.05) * k2).normalized()
    l = (rest_l * (1 - k1 - k2) + D(-0.15, 0.5, 0.85) * k1 + D(1, -0.65, -0.05) * k2).normalized()
    p.arm('R', u, l, palm=PALM_IN, curl=0.9, wrist=-15 * k1 + 20 * k2)
    p.arm('L', mix(ARM_DOWN, D(0.3, 0.7, -0.3), k1 + k2 * 0.6), mix(fwd(14) @ ARM_DOWN, D(0.5, 0.5, -0.2), k1 + k2 * 0.6), palm=PALM_DOWN, curl=0.1)
    return p


@clip(0.45)
def Grab(t):
    """Se lanza a agarrar: manos abiertas por delante y se cierran al atrapar."""
    k = K(t, (0, 0), (0.45, 1), (1, 0))
    close = K(t, (0.35, 0), (0.55, 1), (1, 0.5))
    p = upper().torso(spine=(24 * k, 0, 0), chest=(14 * k, 0, 0), head=(-16 * k, 0, 0))
    for side in 'LR':
        p.arm(side, mix(ARM_DOWN, D(1, 0.3, -0.5), k), mix(fwd(14) @ ARM_DOWN, D(1, 0.05, -0.35), k), palm=PALM_DOWN,
              curl=-0.4 + 1.2 * close, protract=10 * k)
    return p


@clip(0.5)
def Push(t):
    """Empujón: recoge los brazos, empuja con las palmas abiertas y rebota."""
    k1 = K(t, (0, 0), (0.28, 1), (0.42, 0))
    k2 = K(t, (0.28, 0), (0.44, 1), (0.6, 0.85), (1, 0))
    p = upper().torso(spine=(-8 * k1 + 16 * k2, 0, 0), chest=(-6 * k1 + 10 * k2, 0, 0), head=(4 * k1 - 6 * k2, 0, 0))
    for side in 'LR':
        # El pecho se inclina al empujar: los brazos suben algo para que las palmas lleguen a la altura del pecho.
        u = (ARM_DOWN * (1 - k1 - k2) + D(0.3, 0.55, -0.25) * k1 + D(1, 0.25, 0.45) * k2).normalized()
        l = (fwd(14) @ ARM_DOWN * (1 - k1 - k2) + D(0.9, -0.35, 0.45) * k1 + D(1, 0.1, 0.5) * k2).normalized()
        p.arm(side, u, l, palm=mix(PALM_IN, PALM_DOWN, min(1.0, k1 + k2)), curl=-0.35 * k2 + 0.3 * k1, wrist=-60 * k2, protract=14 * k2)
    return p


@clip(0.5, True)
def Hammer(t):
    """Martillazos: sube, golpea con un pequeño rebote; la izquierda sujeta la tabla."""
    k = K(t, (0, 0.05), (0.4, 1), (0.55, -0.08), (0.65, 0.02), (1, 0.05))
    p = upper().torso(spine=(15, 0, 0), chest=(8 - 7 * k, 4 * k, 0), head=(-6 + 3 * k, 0, 0))
    p.arm('R', (D(0.75, 0.25, -0.35) * (1 - k) + D(0.2, 0.6, 0.8) * k).normalized(),
          (D(1, -0.2, -0.45) * (1 - k) + D(0.15, 0.25, 1) * k).normalized(), palm=PALM_IN, curl=0.95, wrist=-20 * k + 10 * (1 - k))
    p.arm('L', D(0.8, -0.05, -0.45), D(0.9, -0.45, -0.25), palm=PALM_DOWN, curl=0.2)
    return p


GUN_HIP = (D(0.2, 0.45, -0.9), D(0.8, -0.3, -0.5))
GUN_AIM_R = (D(0.7, 0.6, -0.15), D(0.85, -0.5, 0.15))
GUN_AIM_L = (D(0.9, -0.1, -0.3), D(0.9, -0.4, 0.05))       # la izquierda sujeta el cañón por delante


def gun(p, k, recoil=0.0):
    p.arm('R', mix(GUN_HIP[0], GUN_AIM_R[0] + D(0, 0, 0.4 * recoil), k), mix(GUN_HIP[1], GUN_AIM_R[1] + D(0, 0, 0.7 * recoil), k),
          palm=PALM_IN, curl=0.95, shrug=4 * k - 6 * recoil, protract=-10 * recoil)
    p.arm('L', mix(ARM_DOWN, GUN_AIM_L[0], k), mix(fwd(14) @ ARM_DOWN, GUN_AIM_L[1], k), palm=PALM_UP, curl=0.7 * k + 0.3)
    return p


@clip(0.65)
def GunDraw(t):
    """Echa la mano atrás por encima del hombro, descuelga la escopeta y la lleva al hombro."""
    grab = K(t, (0, 0), (0.3, 1), (0.48, 1), (1, 0))
    k = K(t, (0.3, 0), (0.85, 1), (1, 1))
    p = upper().torso(chest=(0, -22 * grab + 5 * k, -4 * grab), head=(-4 * k, 6 * grab - 5 * k, 0))
    reach = (D(0.15, 0.8, 0.45), D(-0.3, -0.85, 0.35))   # codo a la altura del hombro, mano por encima del hombro
    p.arm('R', (GUN_HIP[0] * (1 - grab - k) + reach[0] * grab + GUN_AIM_R[0] * k).normalized(),
          (GUN_HIP[1] * (1 - grab - k) + reach[1] * grab + GUN_AIM_R[1] * k).normalized(), palm=PALM_IN, curl=0.95 * max(k, grab))
    p.arm('L', mix(ARM_DOWN, GUN_AIM_L[0], k), mix(fwd(14) @ ARM_DOWN, GUN_AIM_L[1], k), palm=PALM_UP, curl=0.3 + 0.4 * k)
    return p


@clip(1.6, True)
def GunAim(t):
    """Apunta: mejilla cerca de la culata, respiración que mece un poco el cañón."""
    p = upper().torso(chest=(S(t, 0.8), 6 + S(t + 0.3, 0.8), 0), neck=(0, -4, 0), head=(-2, -6, -6))
    return gun(p, 1)


@clip(0.4)
def GunFire(t):
    """Disparo: retroceso que empuja hombro y pecho atrás, la cabeza da un respingo y vuelve a apuntar."""
    k = K(t, (0, 0), (0.12, 1), (0.45, 0.25), (1, 0))
    p = upper().torso(spine=(-7 * k, 0, 0), chest=(-9 * k, 6 - 3 * k, 0), neck=(0, -4, 0), head=(-2 - 8 * k, -6, -6 + 3 * k))
    return gun(p, 1, recoil=k)


@clip(0.45)
def Pecked(t):
    """¡Picotazo!: encoge los hombros, se protege con las manos y aparta la cabeza."""
    k = K(t, (0, 0), (0.18, 1), (0.6, 0.6), (1, 0))
    p = upper().torso(spine=(-10 * k, 0, 0), chest=(-8 * k, 6 * k, 0), neck=(0, 6 * k, 0), head=(-14 * k, 14 * k, 8 * k))
    for side in 'LR':
        p.arm(side, mix(ARM_DOWN, D(0.5, 0.6, 0.35), k), mix(fwd(14) @ ARM_DOWN, D(0.35, -0.25, 0.9), k), palm=PALM_FWD,
              curl=-0.3 * k, shrug=14 * k)
    return p


@clip(1.2, True)
def HoneyCarry(t):
    """Lleva el tarro de miel con la izquierda, delante y con cuidado; la derecha se balancea suelta."""
    p = upper().torso(chest=(0, 4, 0), head=(8, 4 + S(t, 2), 0))
    p.arm('L', D(0.45, 0.35, -0.8), D(0.95, -0.2, 0.12 + S(t, 0.03)), palm=PALM_UP, curl=0.7, wrist=6)
    relaxed_arm(p, 'R', 4 + S(t - 0.1, 6), 18, curl=0.35)
    return p


UPPER = ['Carry', 'Throw', 'NetSwing', 'Grab', 'Push', 'Hammer', 'GunDraw', 'GunAim', 'GunFire', 'Pecked', 'HoneyCarry']

# ============================================================ 6c. PRIMERA PERSONA (QA 0.9.3)
# Clips sólo para el "viewmodel" del dueño en 1ª persona (FarmerBody: un esqueleto copia que sólo enseña antebrazos y
# manos). Los demás jugadores nunca los ven. Se diseñan DESDE LA CÁMARA: un punto de la pantalla (16:9, 0-1) y una
# profundidad para el agarre de cada mano, y la orientación de la herramienta respecto a la pose en la que el juego la
# monta (SceneBuilder: red en Idle, escopeta en GunAim, martillo en el punto más bajo de Hammer, tarro en HoneyCarry).
# El cuerpo (que no se ve) se desplaza lo que haga falta para que los brazos chibi lleguen: así el antebrazo entra desde
# abajo/la esquina, lejos de la cámara, y la mano y la herramienta quedan por encima de la barra de botones.
FP_ALTO = {'Paco': 1.89, 'Lola': 1.78, 'AbueloTomas': 1.76, 'Rosa': 1.67, 'Chema': 1.65}   # = FarmerArt.Models (m)
FP_S = FP_ALTO.get(NAME, 1.75) / H                       # metros del juego por unidad del modelo
_HEAD_Z = J['Head'][0].z
FP_EYE = Vector((0.0, sy - 0.06 / FP_S, _HEAD_Z + (H - _HEAD_Z) * 0.38))   # cámara (SceneBuilder.eyeHeight + SetEyeHeight)
FP_SOCKET = FP_EYE + (FWD * 0.52 - UP * 0.46) / FP_S     # HandSocket: la gallina en brazos (FarmerController.SetEyeHeight)
FP_VFOV, FP_ASPECT = 75.0, 16.0 / 9.0                    # CameraRig.baseFov (vertical)
LEFT = -RIGHT


def fp_point(sx, sy_, d):
    """Punto del modelo que se ve en (sx, sy_) de una pantalla 16:9 a d unidades delante de la cámara (mirando de frente)."""
    ty = math.tan(math.radians(FP_VFOV / 2)); tx = ty * FP_ASPECT
    return FP_EYE + FWD * d + RIGHT * ((sx - 0.5) * 2 * tx * d) + UP * ((sy_ - 0.5) * 2 * ty * d)


def fp_screen(pt):
    d = (pt - FP_EYE).dot(FWD)
    ty = math.tan(math.radians(FP_VFOV / 2)); tx = ty * FP_ASPECT
    return (0.5 + (pt - FP_EYE).dot(RIGHT) / (2 * tx * d), 0.5 + (pt - FP_EYE).dot(UP) / (2 * ty * d), d)


def _posed_of(p):
    posed_rot.clear()
    ps = solve(p.rot, p.hips)
    return ps, dict(posed_rot)


def _grip_point(ps, side):
    return ps[f'Grip.{side}'].translation.copy()


# Herramientas en 1ª persona: el juego (FarmerBody) monta la copia de cada herramienta en el agarre (Grip) con estos ejes,
# en el espacio del personaje: "eje" = mango de la red y del martillo, cañón de la escopeta, arriba del tarro; "otro" =
# frente del aro, arriba de la escopeta, cabeza del martillo, frente del tarro. En Unity: (v·RIGHT, v·UP, v·FWD).
# La mano se pone anatómica: el mango pasa por el puño a lo largo del pulgar.
def _n(v):
    return v.normalized()


_NET_AXIS = _n(UP * 0.75 + FWD * 0.5 + RIGHT * 0.25)      # por el lado derecho: el aro no tapa el centro
_GUN_AXIS = _n(FWD + LEFT * 0.22 + UP * 0.22)      # algo hacia arriba: la culata baja, fuera del cuadro
_HAMMER_AXIS = _n(FWD * 0.45 + UP * 0.85 + LEFT * 0.1)
FP_TOOL = {
    'FP_Net': {'R': ('Net', _NET_AXIS, FWD)},
    'FP_Jar': {'R': ('Net', _NET_AXIS, FWD), 'L': ('Jar', UP, FWD)},
    'FP_Gun': {'R': ('Gun', _GUN_AXIS, UP)},
    'FP_Hammer': {'R': ('Hammer', _HAMMER_AXIS, LEFT)},
}


def fp_hand(side, fingers, thumb):
    """Rotación absoluta de la mano: dedos hacia `fingers` y pulgar (eje del puño) hacia `thumb`."""
    hd = (J[f'Hand.{side}'][1] - J[f'Hand.{side}'][0]).normalized()
    return frame_rot(hd, HAND[side]['thumb_dir'], fingers.normalized(), thumb.normalized())


def _corner(side, up=0.78, fwd=0.5, inward=0.3):
    """Antebrazo que entra desde la esquina de abajo de su lado: hacia delante, arriba y un poco al centro."""
    return _n(FWD * fwd + UP * up + (LEFT if side == 'R' else RIGHT) * inward)


_LUA = {s_: (J[f'UpperArm.{s_}'][1] - J[f'UpperArm.{s_}'][0]).length for s_ in 'LR'}
_LFA = {s_: (J[f'LowerArmTwist.{s_}'][1] - J[f'LowerArm.{s_}'][0]).length for s_ in 'LR'}
FP_INFO = {}


def fp_arms(p, want, curl=None, rest_side=None):
    """want = {lado: (punto del agarre (Grip), dirección del antebrazo, rotación absoluta de la mano)}. Mueve el cuerpo
    (cadera; en 1ª persona no se ve) y pone los brazos por IK: el agarre llega a su punto y el antebrazo entra con la
    dirección pedida (el codo queda por debajo del borde de la pantalla). El brazo libre cuelga atrás, fuera del cuadro."""
    curl = curl or {}
    T = Vector((0, 0, 0))
    for it in range(16):
        p.hips = T.copy()
        if rest_side: relaxed_arm(p, rest_side, -35, 25, curl=0.35)
        ps, pr = _posed_of(p)
        c = p.chest_q
        ci = c.inverted()
        adj, n_ = Vector(), 0
        for side, (G, l, Rh) in want.items():
            W = G - Rh @ (J[f'Grip.{side}'][0] - J[f'Hand.{side}'][0])
            E = W - _LFA[side] * l
            v = E - ps[f'UpperArm.{side}'].translation
            adj += v.normalized() * (v.length - _LUA[side]); n_ += 1
            p0 = ARM_REST[side][1]
            p.arm(side, side_vec(ci @ v.normalized(), side), side_vec(ci @ l, side), palm=side_vec(ci @ (Rh @ p0), side),
                  curl=curl.get(side, 0.6), wrist=0.0)
        if n_: T = T + adj / n_ * 0.9
        T = Vector((max(-0.15, min(0.15, T.x)), max(-0.35, min(0.15, T.y)), max(-0.35, min(0.1, T.z))))
    p.hips = T
    if rest_side: relaxed_arm(p, rest_side, -35, 25, curl=0.35)
    ps, pr = _posed_of(p)
    for side, (G, l, Rh) in want.items():
        Dl = Rh @ pr[f'Hand.{side}'].inverted()
        for b_ in ('Hand', 'Fingers1', 'Fingers2', 'Thumb1', 'Thumb2'):
            nm = f'{b_}.{side}'
            if nm in p.rot: p.rot[nm] = Dl @ p.rot[nm]
    if CUR_CLIP[1] == 0.0:
        ps, pr = _posed_of(p)
        FP_INFO[CUR_CLIP[0]] = {s_: (fp_screen(ps[f'Grip.{s_}'].translation), fp_screen(ps[f'LowerArm.{s_}'].translation),
                                     round(T.length, 3)) for s_ in want}
    return p


def fp_base(t):
    return upper().torso(spine=(0, 0, 0), chest=(2 + S(t) * 0.8, 0, 0), head=(0, 0, 0))


def _sway(t, a=0.006):
    return Vector((S(t + 0.2, a * 0.5), 0, S(t, a)))     # respiración: las manos suben y bajan un poco


def _fist(side, l, axis):
    """Puño alrededor de un mango: pulgar a lo largo del mango y dedos siguiendo al antebrazo (la parte perpendicular)."""
    f = l - axis * l.dot(axis)
    if f.length < 1e-3: f = axis.orthogonal()
    return fp_hand(side, f, axis)


def _net_right(t):
    l = _corner('R')
    return fp_point(0.72, 0.22, 0.27) + _sway(t), l, _fist('R', l, _NET_AXIS)


@clip(1.6, True)
def FP_Idle(t):
    """1ª persona, manos vacías: las dos bajas y adelantadas, puños medio cerrados, sólo antebrazos y manos en cuadro."""
    p = fp_base(t)
    want = {}
    for side, x, inward in (('R', 0.70, LEFT), ('L', 0.30, RIGHT)):
        l = _corner(side, up=0.5, fwd=0.7)
        want[side] = (fp_point(x, 0.19, 0.27) + _sway(t + (0.5 if side == 'L' else 0.0)), l, fp_hand(side, l, _n(UP * 0.6 + inward * 0.5)))
    return fp_arms(p, want, curl={'R': 0.45, 'L': 0.45})


@clip(1.6, True)
def FP_Net(t):
    """1ª persona con la red: mano derecha abajo a la derecha, el antebrazo entra por la esquina y la red sale hacia delante."""
    p = fp_base(t)
    return fp_arms(p, {'R': _net_right(t)}, curl={'R': 0.95}, rest_side='L')


@clip(1.6, True)
def FP_Jar(t):
    """1ª persona con el tarro de miel en la izquierda (abajo a la izquierda, derecho) y la red en la derecha."""
    p = fp_base(t)
    lL = _corner('L', up=0.5, fwd=0.7)
    want = {'R': _net_right(t), 'L': (fp_point(0.27, 0.30, 0.39) + _sway(t + 0.5), lL, fp_hand('L', lL, _n(UP * 0.3 + RIGHT)))}
    return fp_arms(p, want, curl={'R': 0.95, 'L': 0.7})


@clip(1.6, True)
def FP_Gun(t):
    """1ª persona con la escopeta en reposo (sin apuntar): la derecha la lleva por la empuñadura abajo a la derecha, el
    cañón hacia delante, algo arriba y hacia el centro (la culata baja, fuera del cuadro); la izquierda, suelta y fuera.
    (Con la izquierda bajo el cañón, su codo y la manga quedaban en medio de la pantalla.)"""
    p = fp_base(t)
    lR = _corner('R', up=0.7, fwd=0.6, inward=0.3)
    G = fp_point(0.66, 0.18, 0.28) + _sway(t, 0.004)
    RR = fp_hand('R', _n(lR - UP * 0.4), _n(UP * 0.8 + _GUN_AXIS * 0.3))        # empuñadura: puño con el pulgar arriba
    return fp_arms(p, {'R': (G, lR, RR)}, curl={'R': 0.95}, rest_side='L')


@clip(0.5, True)
def FP_Hammer(t):
    """1ª persona reparando: martillazos cortos delante a la derecha (sube, golpea con rebote); la cabeza siempre a la vista."""
    p = fp_base(t)
    k = K(t, (0, 0.05), (0.4, 1), (0.55, -0.08), (0.65, 0.02), (1, 0.05))      # como Hammer: 1 = arriba, 0 = golpe
    axis = (q(LEFT, -40 * (k - 0.05)) @ _HAMMER_AXIS).normalized()              # al levantar, el mango se echa atrás
    l = _corner('R', up=0.95 + 0.1 * k, fwd=0.4, inward=0.35)
    G = fp_point(0.67, 0.23 + 0.07 * k, 0.27)
    return fp_arms(p, {'R': (G, l, _fist('R', l, axis))}, curl={'R': 0.95}, rest_side='L')


@clip(1.0, True)
def FP_Carry(t):
    """1ª persona con una gallina en brazos (va en el HandSocket): las manos a sus lados, por la parte que se ve."""
    p = fp_base(t)
    j = S(t * 2, 0.004)
    want = {}
    for side, sd, inward in (('R', RIGHT, LEFT), ('L', LEFT, RIGHT)):
        l = _corner(side, up=0.35, fwd=0.8, inward=0.45)
        G = FP_SOCKET + sd * (0.15 / FP_S) + UP * (0.14 / FP_S) + Vector((0, 0, j if side == 'R' else -j))
        want[side] = (G, l, fp_hand(side, l, _n(UP + inward * 0.2)))
    return fp_arms(p, want, curl={'R': 0.5, 'L': 0.5})


FP_CLIPS = ['FP_Idle', 'FP_Net', 'FP_Jar', 'FP_Gun', 'FP_Hammer', 'FP_Carry']

# ============================================================ 6b. manos fuera del cuerpo (revisión 0.9.3)
# Las poses son direcciones pensadas para todos los granjeros; un torso más ancho (el chaleco de Lola) se tragaba las manos
# (Run: el puño que sube por delante; Taunt/Dance: manos en la cadera). Perfil del torso de ESTE personaje en reposo (sin
# brazos ni cabeza): por cada altura, una superelipse (|x|/rx)^3 + (|y-cy|/ry)^3 = 1. En cada fotograma, si la mano o el
# antebrazo quedan dentro (en el marco del pecho), se gira el brazo entero sobre el hombro hacia fuera lo justo.
_tm = (~ARM_MASK) & (~head_only) & (CO[:, 2] < cut - band)
_zs = np.arange(crotch, neckZ, 0.02)
TORSO = []
for z0_ in _zs:
    m_ = _tm & (CO[:, 2] >= z0_ - 0.01) & (CO[:, 2] < z0_ + 0.01)
    if m_.sum() < 8: continue
    P_ = CO[m_]
    TORSO.append((float(z0_), float(np.abs(P_[:, 0]).max()), float((P_[:, 1].min() + P_[:, 1].max()) / 2),
                  float((P_[:, 1].max() - P_[:, 1].min()) / 2)))
TORSO = np.array(TORSO).reshape(-1, 4)
# Puntos de control (hueso, extremo, margen = medio grosor de la mano/antebrazo alrededor del eje del hueso)
_hp = _prof[(_prof[:, 0] > ARM['W'][0] + 0.01) & (_prof[:, 0] < ARM['T'][0] - 0.03)]
_HR = 0.75 * float(np.median(_hp[:, 4])) if len(_hp) else float('nan')
if not (math.isfinite(_HR) and _HR > 0): _HR = 0.6 * float(ARM['r_wr'])      # mano corta: radio de la muñeca
CTRL = [('LowerArmTwist', 0, 0.8 * ARM['r_wr']), ('Hand', 0, _HR), ('Fingers1', 0, _HR), ('Fingers2', 1, 0.8 * _HR),
        ('Thumb2', 1, 0.6 * _HR)]
CHAIN = ['UpperArm', 'Elbow', 'LowerArm', 'LowerArmTwist', 'Hand', 'Fingers1', 'Fingers2', 'Thumb1', 'Thumb2']
# Manos que van sobre la herramienta con la otra (la izquierda en el cañón de la escopeta): no se tocan.
# Clips con agarre a dos manos de la escopeta (la derecha la empuña en Grip.R y la izquierda sujeta el cañón): no se toca
# NINGUNA mano, o se rompería la relación entre las dos y el arma. (La gallina de Carry va en el HandSocket del
# controlador, no en las manos, y el tarro de HoneyCarry en Grip.L con una sola mano: ahí sí se corrige.)
NO_TOCAR = {(c_, s_) for c_ in ('GunAim', 'GunFire', 'GunDraw') for s_ in 'LR'}
# GunDraw: mientras la derecha va a por el arma (t < 0.4) la izquierda aún cuelga lejos del cañón: ahí sí se corrige.
LIBRE_HASTA = {'GunDraw': 0.4}
USAR_CHOQUES = FIX['sin_choques'] and len(TORSO) >= 3
if FIX['sin_choques'] and not USAR_CHOQUES: print('TORSO: perfil insuficiente, sin corrección de manos')
CHOQUES = {}
if len(TORSO): print(f'TORSO: perfil en {len(TORSO)} alturas ({TORSO[0, 0]:.2f}-{TORSO[-1, 0]:.2f}), margen mano {_HR:.3f}')


def _dentro(pr):
    """Penetración (0 = fuera) de un punto en reposo del torso; devuelve (prof, dirección hacia fuera en reposo)."""
    x_, y_, z_ = pr
    if z_ < TORSO[0, 0] - 0.01 or z_ > TORSO[-1, 0] + 0.01: return 0.0, None
    rx = float(np.interp(z_, TORSO[:, 0], TORSO[:, 1])); cy = float(np.interp(z_, TORSO[:, 0], TORSO[:, 2]))
    ry = float(np.interp(z_, TORSO[:, 0], TORSO[:, 3]))
    return rx, cy, ry


def sin_choques(p, posed):
    for side in 'LR':
        if f'Hand.{side}' not in p.rot: continue
        if (CUR_CLIP[0], side) in NO_TOCAR and CUR_CLIP[1] >= LIBRE_HASTA.get(CUR_CLIP[0], -1.0): continue
        total = 0.0
        for it in range(20):
            M = posed['Spine1'] @ REST['Spine1'].inverted()          # movimiento del tronco (reposo → pose)
            Mi = M.inverted()
            worst, push = 0.0, None
            for bn, end, mg in CTRL:
                nm = f'{bn}.{side}'
                if nm not in posed: continue
                pt_rest = J[nm][end]
                pw = posed[nm] @ (REST[nm].inverted() @ pt_rest)
                pr = Mi @ pw
                r_ = _dentro(pr)
                if r_[1] is None: continue
                rx, cy, ry = r_
                ax, ay = rx + mg, ry + mg
                u_ = (abs(pr.x) / ax) ** 3 + (abs(pr.y - cy) / ay) ** 3
                if u_ >= 1.0: continue
                dep = (1 - u_ ** (1 / 3)) * min(ax, ay)
                if dep > worst:
                    # Hacia fuera: normal de la superelipse en ese punto
                    nx = (abs(pr.x) / ax) ** 2 / ax * (1 if pr.x >= 0 else -1)
                    ny = (abs(pr.y - cy) / ay) ** 2 / ay * (1 if pr.y >= cy else -1)
                    worst, push = dep, (pw, (M.to_3x3() @ Vector((nx, ny, 0))).normalized())
            if worst < 0.002 or push is None: break
            sh = posed[f'UpperArm.{side}'].translation
            pw, dw = push
            ax_ = (pw - sh).cross(dw)
            if ax_.length < 1e-6: break
            ang = min(4.0, max(1.0, math.degrees(worst / max(0.05, (pw - sh).length))))
            R_ = Quaternion(ax_.normalized(), math.radians(ang))
            for bn in CHAIN:
                nm = f'{bn}.{side}'
                if nm in p.rot: p.rot[nm] = R_ @ p.rot[nm]
            if f'Deltoid.{side}' in p.rot: p.rot[f'Deltoid.{side}'] = qslerp(Quaternion(), R_, 0.5) @ p.rot[f'Deltoid.{side}']
            total += ang
            posed_rot.clear()
            posed = solve(p.rot, p.hips)
        if total > 0:
            st = CHOQUES.setdefault(CUR_CLIP[0], [set(), 0.0])
            st[0].add(round(CUR_CLIP[1], 4)); st[1] = max(st[1], total)
    return p


# ============================================================ 7. hornear acciones
arm_obj.animation_data_create()
for b in pb: b.rotation_mode = 'QUATERNION'
actions = []
for name, (sec, loop, fn) in CLIPS.items():
    act = bpy.data.actions.new(name)
    CUR_CLIP[0] = name
    arm_obj.animation_data.action = act
    n = max(2, int(round(sec * FPS)))
    for f in range(n + 1):
        t = f / n
        if loop and f == n: t = 0.0
        CUR_CLIP[1] = t
        p = fn(t)
        posed_rot.clear()
        posed = solve(p.rot, p.hips)
        if USAR_CHOQUES and not name.startswith('FP_'):      # 1ª persona: el cuerpo no se ve, sólo antebrazos
            p = sin_choques(p, posed)
            posed_rot.clear()
            solve(p.rot, p.hips)
        for b in pb:
            b.keyframe_insert('rotation_quaternion', frame=f, group=b.name)
            if b.name == 'Hips': b.keyframe_insert('location', frame=f, group=b.name)
    act.use_fake_user = True
    if name in ('Walk', 'Run'):
        # Bucle: tangentes cíclicas (sin el modificador, las claves de los extremos quedan planas y el pie apoyado se
        # frena en la costura del bucle).
        fcs = list(getattr(act, 'fcurves', []) or []) or [fc for ly in act.layers for st_ in ly.strips for cb in st_.channelbags for fc in cb.fcurves]
        for fc in fcs:
            for kp in fc.keyframe_points:                # tangentes suaves sin recorte (como las "smooth" de Unity):
                kp.handle_left_type = kp.handle_right_type = 'AUTO'     # el recorte en extremos hace dientes
            fc.modifiers.new('CYCLES')
            fc.update()
    actions.append(act)
    st = TWIST_STATS.get(name, [0, 0, 0])
    print(f'CLIP {name}: {n} fotogramas{" (bucle)" if loop else ""}  giro antebrazo máx {st[0]:.0f}°, giro brazo {st[1]:.0f}°, codo {st[2]:.0f}°'
          f'{"  ¡CODO AL REVÉS " + str(sorted(WARN[name])) + "!" if name in WARN else ""}'
          f'{"  manos fuera del cuerpo en %d fotogramas (máx %.0f°)" % (len(CHOQUES[name][0]), CHOQUES[name][1]) if name in CHOQUES else ""}')

for k_, v_ in FP_INFO.items():
    print(f'FP {k_}: ' + ' | '.join(f'{s_}: agarre en pantalla ({a_[0]:.2f}, {a_[1]:.2f}) a {a_[2] * FP_S:.2f} m, codo en ({e_[0]:.2f}, {e_[1]:.2f}); cuerpo movido {d_ * FP_S:.2f} m'
                                    for s_, (a_, e_, d_) in v_.items()))
for k_, v_ in FP_TOOL.items():
    for s_, (tool_, ax_, ot_) in v_.items():
        print(f'FP_TOOL {k_} {s_} {tool_}: eje Unity ({ax_.dot(RIGHT):.3f}, {ax_.dot(UP):.3f}, {ax_.dot(FWD):.3f}) otro ({ot_.dot(RIGHT):.3f}, {ot_.dot(UP):.3f}, {ot_.dot(FWD):.3f})')

# Pistas NLA silenciadas (para revisar en Blender); el FBX exporta cada acción como una toma (use_all_actions).
arm_obj.animation_data.action = None
for act in actions:
    tr = arm_obj.animation_data.nla_tracks.new()
    tr.name = act.name
    tr.strips.new(act.name, 0, act)
    tr.mute = True

# ============================================================ 8. exportar
for b in pb:
    b.matrix_basis = Matrix.Identity(4)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + '_rig.blend'))
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True); arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, NAME + '.fbx'), use_selection=True, object_types={'ARMATURE', 'MESH'},
                         apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
                         add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
                         bake_anim_use_all_bones=True, bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
                         primary_bone_axis='Y', secondary_bone_axis='X', mesh_smooth_type='FACE')
print('EXPORT', os.path.join(OUT, NAME + '.fbx'), 'clips', len(actions), 'brazos de capa superior', UPPER)
