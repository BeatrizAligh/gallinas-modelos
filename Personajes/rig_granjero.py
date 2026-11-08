"""
Rig y animaciones de un granjero (personajes chibi en pose T, mirando a -Y, Z arriba).
blender -b -P rig_granjero.py -- <retopo.fbx> <textura.png|-> <carpeta_salida> <Nombre> [<alta.fbx para hornear>]

1. Importa el retopo y lo deja en coordenadas de mundo.
2. (Sin UVs) UV automático + horneado del color desde la versión de alta (su textura empaquetada).
3. Mide la malla (hombros, codos, muñecas, cuello, caderas, rodillas, tobillos, puntera) y crea el esqueleto.
4. Pesos automáticos (calor de huesos) y relleno de lo que quede sin peso con el vecino más cercano.
5. Anima con FK por direcciones (cuerpo: adelante -Y, derecha del personaje -X, arriba +Z), clips en sitio a 30 fps.
6. Exporta <Nombre>.fbx (tomas "Granjero|Clip"), <Nombre>.png, <Nombre>_rig.blend y una hoja de poses.
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

# ============================================================ 4. esqueleto
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
s1 = hipZ + 0.06 * H
s2 = (s1 + armBot) / 2
bone('Spine', (0, sy, s1), (0, sy, s2), 'Hips', connect=True)
bone('Chest', (0, sy, s2), (0, sy, neckZ - 0.01 * H), 'Spine', connect=True)
bone('Neck', (0, sy, neckZ - 0.01 * H), (0, sy, neckZ + 0.03 * H), 'Chest', connect=True)
bone('Head', (0, sy, neckZ + 0.03 * H), (0, sy, H * 0.98), 'Neck', connect=True)
for side, sgn, tip in (('L', 1, tipL), ('R', -1, tipR)):
    wrist = tip - handLen
    elbow = shX + 0.5 * (wrist - shX)
    bone(f'Shoulder.{side}', (sgn * torso * 0.25, sy, armZ + 0.01 * H), (sgn * shX, sy, armZ), 'Chest')
    bone(f'UpperArm.{side}', (sgn * shX, sy, armZ), (sgn * elbow, sy, armZ), f'Shoulder.{side}', connect=True)
    bone(f'LowerArm.{side}', (sgn * elbow, sy, armZ), (sgn * wrist, sy, armZ), f'UpperArm.{side}', connect=True)
    bone(f'Hand.{side}', (sgn * wrist, sy, armZ), (sgn * tip, sy, armZ), f'LowerArm.{side}', connect=True)
    # Agarre de las herramientas: en el centro del puño, apuntando arriba (lado del pulgar) en la pose T.
    palm = wrist + 0.45 * handLen
    bone(f'Grip.{side}', (sgn * palm, sy, armZ), (sgn * palm, sy, armZ + 0.06 * H), f'Hand.{side}', deform=False)
    bone(f'UpperLeg.{side}', (sgn * legX, sy, hipZ), (sgn * legX, sy, kneeZ), 'Hips')
    bone(f'LowerLeg.{side}', (sgn * legX, sy, kneeZ), (sgn * legX, sy, ankleZ), f'UpperLeg.{side}', connect=True)
    bone(f'Foot.{side}', (sgn * legX, sy, ankleZ), (sgn * legX, toeY * 0.55 + sy * 0.45, 0.02 * H), f'LowerLeg.{side}', connect=True)
    bone(f'Toe.{side}', (sgn * legX, toeY * 0.55 + sy * 0.45, 0.02 * H), (sgn * legX, toeY, 0.015 * H), f'Foot.{side}', connect=True)
bpy.ops.object.mode_set(mode='OBJECT')

# ============================================================ 5. pesos
select_only(body)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
try:
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
except Exception as e:
    print('PESOS: calor de huesos falló, envolventes:', e)
    bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
groups = {g.index: g.name for g in body.vertex_groups}
weighted = [i for i, v in enumerate(body.data.vertices) if sum(g.weight for g in v.groups) > 0.01]
if len(weighted) < 0.5 * len(body.data.vertices):
    # Plan B: pesos por distancia a los segmentos de los huesos (los dos más cercanos, sin cruzar de lado).
    print(f'PESOS: calor de huesos insuficiente ({len(weighted)}), pesos por distancia')
    deform = [b for b in arm_obj.data.bones if b.use_deform]
    segs = [(b.name, np.array(b.head_local[:]), np.array(b.tail_local[:])) for b in deform]
    for vg in list(body.vertex_groups): body.vertex_groups.remove(vg)
    vgs = {n: body.vertex_groups.new(name=n) for n, _, _ in segs}
    co = np.array([v.co[:] for v in body.data.vertices])
    dist = np.zeros((len(co), len(segs)))
    for j, (n, h, t) in enumerate(segs):
        ab = t - h
        u = np.clip(((co - h) @ ab) / max(1e-6, ab @ ab), 0, 1)
        d = np.linalg.norm(co - (h + u[:, None] * ab), axis=1)
        side = 1 if n.endswith('.L') else -1 if n.endswith('.R') else 0
        if side: d = d + np.where(co[:, 0] * side < -0.01, 1.0, 0.0)   # no pintar el lado contrario
        dist[:, j] = d
    for i in range(len(co)):
        order = np.argsort(dist[i])[:2]
        w = 1.0 / np.maximum(dist[i][order], 1e-3) ** 4
        w /= w.sum()
        for j, wj in zip(order, w):
            if wj > 0.02: vgs[segs[j][0]].add([i], float(wj), 'REPLACE')
    weighted = list(range(len(co)))
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
# Cabeza, pelo y sombrero (todo lo que queda por encima del cuello) siguen sólo a la cabeza: el ala no se dobla con los hombros.
head_vg = body.vertex_groups.get('Head')
cut = neckZ + 0.03 * H
for v in body.data.vertices:
    if v.co.z <= cut: continue
    for g in list(v.groups): body.vertex_groups[g.group].remove([v.index])
    head_vg.add([v.index], 1.0, 'REPLACE')
select_only(body)
bpy.ops.object.vertex_group_limit_total(limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)
print(f'PESOS: {len(weighted)} con peso, {missing} rellenados por vecindad')

# ============================================================ 6. FK por direcciones
pb = arm_obj.pose.bones
REST = {b.name: b.matrix_local.copy() for b in arm_obj.data.bones}
FWD, RIGHT, UP = Vector((0, -1, 0)), Vector((-1, 0, 0)), Vector((0, 0, 1))
PITCH, YAW, ROLL = Vector((1, 0, 0)), Vector((0, 0, 1)), Vector((0, -1, 0))   # + = adelante, izquierda, inclinar a la derecha


def q(axis, deg):
    return Quaternion(axis, math.radians(deg))


def D(f=0.0, r=0.0, u=0.0):
    """Dirección en el cuerpo: adelante, derecha (del personaje), arriba."""
    return (FWD * f + RIGHT * r + UP * u).normalized()


def aim(bone_name, direction, base=Quaternion()):
    """Rotación absoluta que lleva el hueso de su dirección de reposo a 'direction' (girada por 'base')."""
    h, t = J[bone_name]
    rest = (t - h).normalized()
    return rest.rotation_difference(direction)


def solve(abs_rot, hips_offset=Vector()):
    """abs_rot: hueso -> rotación absoluta (respecto a la pose T). Fija matrix_basis de cada hueso (FK)."""
    posed = {}
    for b in arm_obj.data.bones:   # padres antes que hijos
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

# ---------------- poses de piezas
ARM_DOWN = {'L': D(0, -0.28, -1), 'R': D(0, 0.28, -1)}


def side_vec(v, side):
    """Espejo izquierda/derecha de una dirección pensada para el brazo derecho."""
    return v if side == 'R' else Vector((-v.x, v.y, v.z))


def body_frame(pitch=0.0, yaw=0.0, roll=0.0):
    return q(YAW, yaw) @ q(PITCH, pitch) @ q(ROLL, roll)


class Pose:
    def __init__(self):
        self.rot = {}
        self.hips = Vector()

    def torso(self, hips=(0, 0, 0), spine=(0, 0, 0), chest=(0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0)):
        bh = body_frame(*hips)
        bs = bh @ body_frame(*spine)
        bc = bs @ body_frame(*chest)
        bn = bc @ body_frame(*neck)
        bhd = bn @ body_frame(*head)
        self.rot.update(Hips=bh, Spine=bs, Chest=bc, Neck=bn, Head=bhd, Root=Quaternion())
        self.chest_q, self.hips_q = bc, bh
        return self

    def arm(self, side, upper, lower, hand=None, shoulder=0.0):
        """Direcciones pensadas para el lado derecho, en el marco del pecho (se espejan para el izquierdo)."""
        c = self.chest_q
        su = c @ side_vec(upper, side)
        sl = c @ side_vec(lower, side)
        sh = c @ side_vec(hand if hand is not None else lower, side)
        sho = c @ q(ROLL if side == 'R' else -ROLL, -shoulder) if shoulder else c
        self.rot[f'Shoulder.{side}'] = sho
        self.rot[f'UpperArm.{side}'] = aim(f'UpperArm.{side}', su)
        self.rot[f'LowerArm.{side}'] = aim(f'LowerArm.{side}', sl)
        self.rot[f'Hand.{side}'] = aim(f'Hand.{side}', sh)
        return self

    def leg(self, side, swing=0.0, knee=0.0, foot=0.0, spread=0.0, toe=0.0):
        """swing + = pierna adelante; knee = flexión; foot + = punta arriba."""
        h = self.hips_q
        up_dir = h @ (q(ROLL, spread if side == 'R' else -spread) @ q(PITCH, -swing) @ Vector((0, 0, -1)))
        lo_dir = h @ (q(ROLL, spread if side == 'R' else -spread) @ q(PITCH, -swing + knee) @ Vector((0, 0, -1)))
        fh, ft = J[f'Foot.{side}']
        frest = (ft - fh).normalized()
        f_dir = h @ (q(PITCH, -swing + knee - foot) @ frest)
        th, tt = J[f'Toe.{side}']
        t_dir = h @ (q(PITCH, -swing + knee - foot - toe) @ (tt - th).normalized())
        self.rot[f'UpperLeg.{side}'] = aim(f'UpperLeg.{side}', up_dir)
        self.rot[f'LowerLeg.{side}'] = aim(f'LowerLeg.{side}', lo_dir)
        self.rot[f'Foot.{side}'] = aim(f'Foot.{side}', f_dir)
        self.rot[f'Toe.{side}'] = aim(f'Toe.{side}', t_dir)
        return self


LEG = J['UpperLeg.L'][0].z - J['Foot.L'][0].z   # longitud aproximada de pierna (cadera → tobillo)


def drop_for(knee_deg, swing_deg=0.0):
    """Cuánto baja la cadera al doblar la rodilla (pies en el suelo)."""
    k, s = math.radians(knee_deg), math.radians(swing_deg)
    half = LEG / 2
    return half * (1 - math.cos(s)) + half * (1 - math.cos(k - s))


def S(t, a=1.0):
    return math.sin(2 * math.pi * t) * a


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def K(t, *pts):
    """Interpolación suave entre (tiempo, valor)."""
    if t <= pts[0][0]: return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0))
            if isinstance(v0, tuple): return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return pts[-1][1]


def base_legs(p):
    p.leg('L').leg('R')
    return p


def fwd(deg):
    """Giro hacia delante (para brazos: + = el brazo va adelante/arriba)."""
    return q(PITCH, -deg)


def arms_down(p, swingL=0.0, swingR=0.0, bend=12.0):
    for side, sw in (('L', swingL), ('R', swingR)):
        u = fwd(sw) @ ARM_DOWN['R']
        l = fwd(sw + bend) @ ARM_DOWN['R']
        p.arm(side, u, l)
    return p


# ---------------- clips: (segundos, bucle, función(t normalizado 0..1) -> Pose)
CLIPS = {}


def clip(sec, loop=False):
    def deco(fn):
        CLIPS[fn.__name__] = (sec, loop, fn)
        return fn
    return deco


@clip(3.0, True)
def Idle(t):
    p = Pose().torso(hips=(0, 0, S(t, 1.2)), spine=(S(t, 1.5) + 2, 0, 0), chest=(S(t + 0.1, 1.5), 0, 0),
                     head=(-3 + S(t + 0.2, 2), S(t * 0.5, 6), 0))
    p.hips = Vector((0, 0, S(t * 2, 0.004)))
    base_legs(p)
    return arms_down(p, S(t, 3), S(t + 0.5, 3), 14 + S(t, 3))


@clip(1.0, True)
def Walk(t):
    sw = 26 * math.sin(2 * math.pi * t)
    p = Pose().torso(hips=(4, sw * 0.25, -S(t, 2.5)), spine=(2, -sw * 0.35, 0), chest=(0, -sw * 0.15, 0), head=(-4, sw * 0.2, S(t, 1.5)))
    kneeL = 8 + max(0.0, -math.sin(2 * math.pi * t)) * 32 + max(0, math.sin(2 * math.pi * (t + 0.25))) * 6
    kneeR = 8 + max(0.0, math.sin(2 * math.pi * t)) * 32 + max(0, math.sin(2 * math.pi * (t + 0.75))) * 6
    p.leg('L', sw, kneeL, foot=max(0, sw) * 0.4 - max(0, -sw) * 0.3).leg('R', -sw, kneeR, foot=max(0, -sw) * 0.4 - max(0, sw) * 0.3)
    p.hips = Vector((0, 0, -drop_for(8) + abs(math.cos(2 * math.pi * t)) * 0.012 * H))
    return arms_down(p, -sw * 0.8, sw * 0.8, 22)


@clip(0.62, True)
def Run(t):
    sw = 40 * math.sin(2 * math.pi * t)
    p = Pose().torso(hips=(12, sw * 0.3, -S(t, 3)), spine=(5, -sw * 0.4, 0), chest=(2, -sw * 0.2, 0), head=(-12, sw * 0.25, 0))
    kneeL = 20 + max(0.0, -math.sin(2 * math.pi * t)) * 70
    kneeR = 20 + max(0.0, math.sin(2 * math.pi * t)) * 70
    p.leg('L', sw, kneeL, foot=-10).leg('R', -sw, kneeR, foot=-10)
    p.hips = Vector((0, 0, -drop_for(20) + abs(math.sin(2 * math.pi * t)) * 0.03 * H))
    for side, s in (('L', -sw), ('R', sw)):
        u = fwd(s * 1.1 + 10) @ D(0, 0.2, -1)
        l = fwd(s * 1.1 + 95) @ D(0, 0.1, -1)
        p.arm(side, u, l)
    return p


@clip(0.35)
def Jump(t):
    k = K(t, (0, 0), (0.45, 1), (1, 0))
    up = K(t, (0, 0), (0.5, 0), (1, 1))
    p = Pose().torso(hips=(10 * k, 0, 0), spine=(8 * k, 0, 0), head=(-10 * k, 0, 0))
    p.leg('L', 15 * k - 5 * up, 40 * k + 10 * up, foot=-20 * up).leg('R', 15 * k - 5 * up, 40 * k + 10 * up, foot=-20 * up)
    p.hips = Vector((0, 0, -drop_for(40 * k, 15 * k)))
    for side in 'LR':
        p.arm(side, fwd(-30 * k + 70 * up) @ ARM_DOWN['R'], fwd(-10 * k + 90 * up) @ ARM_DOWN['R'])
    return p


@clip(0.8, True)
def Air(t):
    p = Pose().torso(hips=(-4, 0, 0), spine=(-3, 0, 0), head=(2 + S(t, 2), 0, 0))
    p.leg('L', 25 + S(t, 5), 55, foot=-15).leg('R', 5 - S(t, 5), 35, foot=-15)
    for side, o in (('L', 0), ('R', 0.5)):
        p.arm(side, fwd(60) @ D(0, 0.6, -0.4), fwd(80 + S(t + o, 8)) @ D(0, 0.5, -0.5))
    return p


@clip(0.35)
def Land(t):
    k = K(t, (0, 0.6), (0.3, 1), (1, 0))
    p = Pose().torso(hips=(12 * k, 0, 0), spine=(10 * k, 0, 0), head=(-12 * k, 0, 0))
    p.leg('L', 20 * k, 55 * k, foot=-10 * k).leg('R', 20 * k, 55 * k, foot=-10 * k)
    p.hips = Vector((0, 0, -drop_for(55 * k, 20 * k)))
    for side in 'LR':
        p.arm(side, fwd(30 * k) @ D(0, 0.8 * k + 0.28, -1), fwd(45 * k) @ D(0, 0.8 * k + 0.28, -1))
    return p


@clip(0.9)
def Slip(t):
    """Resbalón con un huevo: los pies se van hacia delante y cae de culo hacia atrás."""
    k = K(t, (0, 0), (0.35, 1), (1, 1))
    flail = math.sin(t * 30) * (1 - k * 0.6)
    p = Pose().torso(hips=(-50 * k, 0, 8 * flail), spine=(-10 * k, 0, 0), chest=(5 * k, 0, 0), head=(25 * k, 0, 10 * flail))
    p.leg('L', 70 * k, 20 * k, foot=20).leg('R', 50 * k, 40 * k, foot=20)
    p.hips = Vector((0, 0, -LEG * 0.75 * K(t, (0, 0), (0.6, 1), (1, 1))))
    for side, o in (('L', 0), ('R', 1)):
        p.arm(side, fwd(100 * k + 25 * math.sin(t * 25 + o)) @ D(0, 1, 0.2), fwd(120 * k) @ D(0, 0.8, 0.4))
    return p


@clip(1.2, True)
def Lie(t):
    p = Pose().torso(hips=(-50, 0, 0), spine=(-10, 0, 0), chest=(5, 0, 0), head=(25 + S(t, 4), S(t * 0.5, 12), 0))
    p.leg('L', 70, 20, foot=20).leg('R', 50, 40, foot=20)
    p.hips = Vector((0, 0, -LEG * 0.75))
    for side in 'LR':
        p.arm(side, fwd(60) @ D(0, 1, -0.3), fwd(60) @ D(0, 1, -0.2))
    return p


@clip(0.8)
def GetUp(t):
    k = 1 - K(t, (0, 0), (0.85, 1))
    p = Pose().torso(hips=(-50 * k + 25 * math.sin(math.pi * t), 0, 0), spine=(-10 * k + 20 * math.sin(math.pi * t), 0, 0),
                     head=(25 * k - 10 * math.sin(math.pi * t), 0, 0))
    p.leg('L', 70 * k + 30 * math.sin(math.pi * t), 20 * k + 70 * math.sin(math.pi * t), foot=20 * k).leg('R', 50 * k, 40 * k + 30 * math.sin(math.pi * t), foot=20 * k)
    p.hips = Vector((0, 0, -LEG * 0.75 * k))
    for side in 'LR':
        p.arm(side, fwd(60 * k) @ D(0, 1 - 0.72 * (1 - k), -0.3 - 0.7 * (1 - k)), fwd(60 * k + 12) @ D(0, 1 - 0.72 * (1 - k), -0.2 - 0.8 * (1 - k)))
    return p


@clip(1.0, True)
def Stunned(t):
    """Mareado tras un picotazo: tambaleo y cabeza en círculos."""
    p = Pose().torso(hips=(S(t, 4), 0, S(t + 0.25, 6)), spine=(4, 0, S(t + 0.25, 4)), head=(10 + S(t, 8), S(t + 0.25, 15), S(t, 10)))
    p.leg('L', S(t, 6), 12).leg('R', -S(t, 6), 12)
    p.hips = Vector((0, 0, -drop_for(12)))
    for side, o in (('L', 0), ('R', 0.5)):
        p.arm(side, fwd(10 + S(t + o, 10)) @ D(0, 0.55, -1), fwd(30) @ D(0, 0.5, -1))
    return p


@clip(1.0, True)
def Victory(t):
    b = abs(math.sin(2 * math.pi * t))
    p = Pose().torso(hips=(-4, 0, 0), spine=(-6, 0, 0), head=(-15 - b * 5, S(t, 10), 0))
    p.leg('L', 0, 10 + b * 25).leg('R', 0, 10 + b * 25)
    p.hips = Vector((0, 0, -drop_for(10 + b * 25) + b * 0.06 * H))
    for side, o in (('L', 0), ('R', 0.5)):
        p.arm(side, D(0, 0.35, 1), D(S(t + o, 0.2), 0.25, 1))
    return p


@clip(1.4, True)
def Defeat(t):
    p = Pose().torso(hips=(6, 0, 0), spine=(14, 0, 0), chest=(8, 0, 0), head=(25, S(t, 15), 0))
    base_legs(p)
    return arms_down(p, -5, -5, 8)


# ---- capa de brazos (sólo importan pecho, brazos y cabeza; las piernas quedan en reposo)
def upper(t=0.0):
    p = Pose().torso(spine=(0, 0, 0))
    base_legs(p)
    return p


@clip(1.0, True)
def Carry(t):
    """Brazos delante a la altura del pecho, sujetando a la gallina (se revuelve un poco)."""
    p = upper().torso(spine=(-4, 0, 0), chest=(-3, S(t * 2, 3), 0), head=(8, 0, 0))
    base_legs(p)
    for side, o in (('L', 0), ('R', 0.5)):
        p.arm(side, fwd(0) @ D(0.75, 0.35, -0.55), D(0.6, -0.75, 0.25 + S(t * 2 + o, 0.05)), shoulder=4)
    return p


@clip(0.6)
def Throw(t):
    k1, k2 = K(t, (0, 0), (0.35, 1), (0.55, 0)), K(t, (0.35, 0), (0.55, 1), (1, 0))
    p = upper().torso(spine=(-15 * k1 + 25 * k2, 25 * k1 - 30 * k2, 0), chest=(-10 * k1 + 15 * k2, 10 * k1 - 15 * k2, 0), head=(5 * k1 - 10 * k2, 0, 0))
    base_legs(p)
    for side in 'LR':
        u = (D(0.75, 0.35, -0.55) * (1 - k1 - k2) + D(-0.2, 0.3, 1) * k1 + D(1, 0.1, 0.3) * k2).normalized()
        l = (D(0.6, -0.75, 0.25) * (1 - k1 - k2) + D(0.2, 0.0, 1) * k1 + D(1, 0, 0.1) * k2).normalized()
        p.arm(side, u, l)
    return p


@clip(0.6)
def NetSwing(t):
    k1, k2 = K(t, (0, 0), (0.3, 1), (0.5, 0)), K(t, (0.3, 0), (0.5, 1), (1, 0))
    p = upper().torso(chest=(5 * k2, 30 * k1 - 45 * k2, 0), spine=(8 * k2, 10 * k1 - 15 * k2, 0))
    base_legs(p)
    rest = (fwd(0) @ ARM_DOWN['R'], fwd(12) @ ARM_DOWN['R'])
    u = (rest[0] * (1 - k1 - k2) + D(-0.3, 0.7, 0.7) * k1 + D(1, -0.4, 0.1) * k2).normalized()
    l = (rest[1] * (1 - k1 - k2) + D(-0.1, 0.5, 0.9) * k1 + D(1, -0.6, 0.0) * k2).normalized()
    p.arm('R', u, l)
    p.arm('L', fwd(20 * k2) @ ARM_DOWN['R'], fwd(40 * k2) @ ARM_DOWN['R'])
    return p


@clip(0.4)
def Grab(t):
    k = K(t, (0, 0), (0.45, 1), (1, 0))
    p = upper().torso(spine=(25 * k, 0, 0), chest=(15 * k, 0, 0), head=(-15 * k, 0, 0))
    base_legs(p)
    for side in 'LR':
        p.arm(side, (ARM_DOWN['R'] * (1 - k) + D(1, 0.25, -0.6) * k).normalized(), (ARM_DOWN['R'] * (1 - k) + D(1, 0.1, -0.5) * k).normalized())
    return p


@clip(0.5)
def Push(t):
    k1, k2 = K(t, (0, 0), (0.3, 1), (0.5, 0)), K(t, (0.3, 0), (0.45, 1), (1, 0))
    p = upper().torso(spine=(-8 * k1 + 15 * k2, 0, 0), chest=(-5 * k1 + 10 * k2, 0, 0))
    base_legs(p)
    for side in 'LR':
        u = (ARM_DOWN['R'] * (1 - k1 - k2) + D(0.4, 0.5, -0.2) * k1 + D(1, 0.2, 0.15) * k2).normalized()
        l = (ARM_DOWN['R'] * (1 - k1 - k2) + D(0.9, -0.3, 0.4) * k1 + D(1, 0.1, 0.2) * k2).normalized()
        p.arm(side, u, l)
    return p


@clip(0.5, True)
def Hammer(t):
    """Martillazos con la derecha; la izquierda sujeta la tabla."""
    k = K(t, (0, 0), (0.45, 1), (0.6, 0), (1, 0))
    p = upper().torso(spine=(15, 0, 0), chest=(8 - 6 * k, 0, 0), head=(-5, 0, 0))
    base_legs(p)
    p.arm('R', (D(0.7, 0.25, -0.3) * (1 - k) + D(0.3, 0.3, 0.9) * k).normalized(), (D(1, -0.2, -0.4) * (1 - k) + D(0.2, -0.3, 0.95) * k).normalized())
    p.arm('L', D(0.8, -0.1, -0.4), D(0.9, -0.5, -0.2))
    return p


GUN_HIP = (D(0.2, 0.5, -0.9), D(0.8, -0.3, -0.5))
GUN_AIM_R = (D(0.75, 0.55, -0.1), D(0.85, -0.5, 0.15))
GUN_AIM_L = (D(0.9, -0.25, -0.2), D(0.6, -0.8, 0.0))


def gun(p, k):
    p.arm('R', (GUN_HIP[0] * (1 - k) + GUN_AIM_R[0] * k).normalized(), (GUN_HIP[1] * (1 - k) + GUN_AIM_R[1] * k).normalized(), shoulder=4 * k)
    p.arm('L', (ARM_DOWN['R'] * (1 - k) + GUN_AIM_L[0] * k).normalized(), (fwd(12) @ ARM_DOWN['R'] * (1 - k) + GUN_AIM_L[1] * k).normalized())
    return p


@clip(0.65)
def GunDraw(t):
    """Descuelga la escopeta de la espalda y se la lleva al hombro."""
    grab = K(t, (0, 0), (0.3, 1), (0.5, 1), (1, 0))
    k = K(t, (0.3, 0), (1, 1))
    p = upper().torso(chest=(0, -20 * grab + 5 * k, 0), head=(-4 * k, 0, 0))
    base_legs(p)
    reach = (D(-0.4, 0.4, 0.8), D(-0.8, -0.1, 0.6))
    p.arm('R', (GUN_HIP[0] * (1 - grab - k) + reach[0] * grab + GUN_AIM_R[0] * k).normalized(),
          (GUN_HIP[1] * (1 - grab - k) + reach[1] * grab + GUN_AIM_R[1] * k).normalized())
    p.arm('L', (ARM_DOWN['R'] * (1 - k) + GUN_AIM_L[0] * k).normalized(), (fwd(12) @ ARM_DOWN['R'] * (1 - k) + GUN_AIM_L[1] * k).normalized())
    return p


@clip(1.6, True)
def GunAim(t):
    p = upper().torso(chest=(S(t, 1), 5 + S(t + 0.3, 1), 0), head=(-4, -5, 0))
    base_legs(p)
    return gun(p, 1)


@clip(0.35)
def GunFire(t):
    k = K(t, (0, 0), (0.15, 1), (1, 0))
    p = upper().torso(spine=(-6 * k, 0, 0), chest=(-8 * k, 5, 0), head=(-4 - 6 * k, -5, 0))
    base_legs(p)
    gun(p, 1)
    p.arm('R', (GUN_AIM_R[0] + D(0, 0, 0.5 * k)).normalized(), (GUN_AIM_R[1] + D(0, 0, 0.8 * k)).normalized())
    return p


@clip(0.4)
def Pecked(t):
    k = K(t, (0, 0), (0.25, 1), (1, 0))
    p = upper().torso(spine=(-10 * k, 0, 0), chest=(-8 * k, 0, 0), head=(-15 * k, 10 * k, 0))
    base_legs(p)
    for side in 'LR':
        p.arm(side, (ARM_DOWN['R'] * (1 - k) + D(0.5, 0.6, 0.4) * k).normalized(), (ARM_DOWN['R'] * (1 - k) + D(0.3, -0.2, 0.9) * k).normalized())
    return p


@clip(1.2, True)
def HoneyCarry(t):
    """Tarro de miel en la izquierda, delante, con cuidado."""
    p = upper().torso(head=(6, 4, 0))
    base_legs(p)
    p.arm('L', D(0.5, 0.3, -0.8), D(0.95, -0.2, 0.15 + S(t, 0.04)))
    p.arm('R', fwd(5) @ ARM_DOWN['R'], fwd(18) @ ARM_DOWN['R'])
    return p


UPPER = ['Carry', 'Throw', 'NetSwing', 'Grab', 'Push', 'Hammer', 'GunDraw', 'GunAim', 'GunFire', 'Pecked', 'HoneyCarry']

# ============================================================ 7. hornear acciones
arm_obj.animation_data_create()
for b in pb: b.rotation_mode = 'QUATERNION'
actions = []
for name, (sec, loop, fn) in CLIPS.items():
    act = bpy.data.actions.new(name)
    arm_obj.animation_data.action = act
    n = max(2, int(round(sec * FPS)))
    for f in range(n + 1):
        t = f / n
        if loop and f == n: t = 0.0
        p = fn(t)
        posed_rot.clear()
        solve(p.rot, p.hips)
        for b in pb:
            b.keyframe_insert('rotation_quaternion', frame=f, group=b.name)
            if b.name == 'Hips': b.keyframe_insert('location', frame=f, group=b.name)
    act.use_fake_user = True
    actions.append(act)
    print(f'CLIP {name}: {n} fotogramas{" (bucle)" if loop else ""}')

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
