"""Rig y animación Idle de las HOJAS de los 5 manzanos (tronco, hierba y base quietos).

blender -b -P rig_manzanos.py            -> <Nombre>_anim.blend y <Nombre>_anim.fbx por cada FBX original
blender -b -P rig_manzanos.py -- pesos   -> además, renders del reparto de pesos en _pesos/

Cómo se decide qué se mueve (la malla viene sin materiales ni UV, así que es por geometría):
  * Piezas sueltas que no tocan el suelo (bolas de copa, hojas, manzanas) = copa.
  * La pieza que toca el suelo (tronco, base, hierba) queda quieta, salvo en los manzanos donde la copa está
    soldada al tronco: ahí el peso sube con la altura, empezando en la manzana/hoja suelta más baja.
  * Las piezas pequeñas (manzanas y hojas sueltas) toman todas el peso de su centro: se mueven en bloque, sin deformarse.
  * Los pesos son continuos en el espacio y suman 1 en cada vértice, así que ninguna cara se separa ni se rompe:
    la malla no se toca (mismos vértices y caras que el original).

Huesos: Raiz (tronco, quieto) > Copa (balanceo general, pivote en lo alto del tronco) > Rama_01..N (racimos de hojas).
Idle de 120 fotogramas a 30 fps (4 s) en bucle perfecto: todas las ondas dan un número entero de vueltas.
"""
import bpy, bmesh, os, sys, glob, math, random
import numpy as np
from mathutils import Vector, Matrix

DIR = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
FPS, LOOP = 30, 120
SMALL_ISLAND = 600       # caras: por debajo, pieza rígida (manzana, hoja suelta, ramita)
GROUND_Z = 0.12          # una pieza que baja de aquí "toca el suelo" (tronco)
STATIC_TOP = 0.2         # pieza que no sube de aquí: hierba y base, siempre quietas


def islands_of(bm):
    seen, out = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, isl = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            isl.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        verts = sorted({v.index for g in isl for v in g.verts})
        out.append((len(isl), verts))
    return out


def kmeans(pts, k, seed):
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(40):
        d = ((pts[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            sel = pts[lab == j]
            if len(sel):
                c[j] = sel.mean(0)
    return c


THIN = 0.16              # m: más delgado que esto (por dentro) es tronco o rama; arriba se exige menos grosor
# m, a partir de 15 cm por encima del inicio de la copa. Medido a ojo con los renders de _pesos: en el Manzano4 las
# bolas pequeñas de la copa tocan las ramas y miden menos de 16 cm, así que allí arriba sólo pasan ramas finas.
THIN_TOP = {"Manzano4": 0.11}


def trunk_from_ground(obj, vs, co, z0, top):
    """Todo lo que está por debajo de la copa es tronco; desde ahí crece sólo por vértices delgados (grosor medido con un rayo hacia dentro)."""
    from mathutils.bvhtree import BVHTree
    me = obj.data
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    vset = set(vs)
    thin = {}
    for i in vs:
        v = me.vertices[i]
        n = v.normal
        hit = bvh.ray_cast(v.co - n * 1e-4, -n, 0.6)
        limit = THIN + (top - THIN) * min(1.0, max(0.0, (co[i, 2] - z0) / 0.15))
        thin[i] = hit[0] is not None and hit[3] < limit
    adj = {i: [] for i in vs}
    for e in me.edges:
        a, b = e.vertices
        if a in vset and b in vset:
            adj[a].append(b)
            adj[b].append(a)
    seen = {i for i in vs if co[i, 2] < z0}
    stack = list(seen)
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j not in seen and thin[j]:
                seen.add(j)
                stack.append(j)
    return np.array([i in seen for i in vs])


def smooth_on_mesh(obj, vs, m, iters):
    """Suaviza la máscara por las aristas: la transición dura unos pocos anillos, sin saltos."""
    idx = {v: k for k, v in enumerate(vs)}
    pairs = [(idx[a], idx[b]) for a, b in (e.vertices for e in obj.data.edges) if a in idx and b in idx]
    pa = np.array(pairs)
    deg = np.bincount(pa.ravel(), minlength=len(vs)).astype(float)
    for _ in range(iters):
        acc = np.zeros(len(vs))
        np.add.at(acc, pa[:, 0], m[pa[:, 1]])
        np.add.at(acc, pa[:, 1], m[pa[:, 0]])
        m = np.where(deg > 0, 0.5 * m + 0.5 * acc / np.maximum(deg, 1), m)
    return m


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def build(path, idx):
    name = os.path.splitext(os.path.basename(path))[0].replace(" ", "")
    name = name[0].upper() + name[1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]
    obj.name = name
    obj.data.name = name + "_Mesh"
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)  # Z arriba, escala 1
    nv0, nf0 = len(obj.data.vertices), len(obj.data.polygons)

    co = np.array([v.co[:] for v in obj.data.vertices])
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    isl = islands_of(bm)
    bm.free()

    # ---- Máscara de copa (0 = quieto, 1 = se mueve) ----
    mask = np.zeros(len(co))
    rigid = []                     # piezas pequeñas que se mueven en bloque
    loose_min = []
    for nf, vs in isl:
        z = co[vs, 2]
        if z.min() > GROUND_Z and z.max() >= STATIC_TOP:
            loose_min.append(z.min())
    z0 = min(loose_min)            # la pieza suelta más baja marca dónde empieza la copa
    fused = False
    for nf, vs in isl:
        z = co[vs, 2]
        if z.max() < STATIC_TOP:
            continue               # hierba, base
        if z.min() > GROUND_Z:
            mask[vs] = 1.0
            if nf < SMALL_ISLAND:
                rigid.append(vs)
        elif z.max() > z0 + 0.25:
            # Copa soldada al tronco: el tronco y las ramas (piezas delgadas que se alcanzan desde el suelo sin
            # atravesar nada grueso) quedan quietos; el resto (bolas de hojas y lo que cuelga de ellas) se mueve.
            trunk = trunk_from_ground(obj, vs, co, z0, THIN_TOP.get(name, THIN))
            if (~trunk).sum() > 50:
                fused = True
                m = np.where(trunk, 0.0, 1.0)
                mask[vs] = smooth_on_mesh(obj, vs, m, iters=6)
    foliage = mask > 0.001
    top = co[foliage, 2].max()

    # ---- Huesos ----
    trunk_xy = co[(co[:, 2] > z0 - 0.08) & (co[:, 2] < z0 - 0.02) & ~foliage][:, :2]
    if len(trunk_xy) == 0:
        trunk_xy = co[foliage][:, :2]
    pivot = Vector((*trunk_xy.mean(0), z0))
    fpts = co[foliage & (mask > 0.5)]
    span = np.ptp(fpts, 0)
    k = int(np.clip(round(span[0] * span[1] / 0.07), 4, 7))
    centers = kmeans(fpts, k, seed=idx + 3)
    centers = centers[np.argsort(np.arctan2(centers[:, 1] - pivot.y, centers[:, 0] - pivot.x))]

    arm_data = bpy.data.armatures.new(name + "_Rig")
    arm = bpy.data.objects.new(name + "_Rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    root = arm_data.edit_bones.new("Raiz")
    root.head, root.tail = Vector((pivot.x, pivot.y, 0)), Vector((pivot.x, pivot.y, 0.1))
    copa = arm_data.edit_bones.new("Copa")
    copa.head, copa.tail, copa.parent = pivot, Vector((pivot.x, pivot.y, top)), root
    ramas = []
    for j, c in enumerate(centers):
        c = Vector(c)
        b = arm_data.edit_bones.new(f"Rama_{j + 1:02d}")
        b.head = pivot.lerp(c, 0.45)        # pivote entre el tronco y el racimo: el racimo se mece
        b.tail = c + (c - b.head).normalized() * 0.05
        b.parent = copa
        ramas.append(b.name)
    bpy.ops.object.mode_set(mode='OBJECT')

    # ---- Pesos ----
    d2 = ((co[:, None, :] - centers[None]) ** 2).sum(-1)
    sigma = max(0.08, float(np.sqrt(d2[foliage].min(1)).mean()) * 1.4)
    share = np.exp(-d2 / (2 * sigma * sigma))
    share /= share.sum(1, keepdims=True)
    for vs in rigid:                         # en bloque: el peso de su centro
        cen = co[vs].mean(0)
        s = np.exp(-((centers - cen) ** 2).sum(-1) / (2 * sigma * sigma))
        share[vs] = s / s.sum()
        mask[vs] = 1.0
    w_rama = share * mask[:, None]
    groups = {"Raiz": obj.vertex_groups.new(name="Raiz")}
    for n in ramas:
        groups[n] = obj.vertex_groups.new(name=n)
    for i in range(len(co)):
        if mask[i] < 0.999:
            groups["Raiz"].add([i], float(1 - mask[i]), 'REPLACE')
        for j, n in enumerate(ramas):
            if w_rama[i, j] > 0.002:
                groups[n].add([i], float(w_rama[i, j]), 'REPLACE')
    # Normalizar (lo podado vuelve al conjunto).
    bpy.context.view_layer.objects.active = obj
    for o in bpy.data.objects:
        o.select_set(o == obj)
    bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    obj.parent = arm
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm

    # ---- Idle: balanceo de la copa + racimos desfasados, bucle de 4 s ----
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, LOOP
    act = bpy.data.actions.new("Idle")  # en el FBX sale como <Nombre>_Rig|Idle
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    rnd = random.Random(100 + idx)
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'

    def wave(f, cycles, phase):
        return math.sin(2 * math.pi * (cycles * (f - 1) / LOOP + phase))

    ph = rnd.random()
    keys = {"Copa": lambda f: (math.radians(1.6) * wave(f, 1, ph), math.radians(0.6) * wave(f, 1, ph + 0.3),
                               math.radians(1.1) * wave(f, 1, ph + 0.25))}
    for n in ramas:
        a, b, c = rnd.random(), rnd.random(), rnd.random()
        cyc = rnd.choice([2, 2, 3])
        keys[n] = (lambda a=a, b=b, c=c, cyc=cyc: lambda f: (
            math.radians(2.4) * wave(f, cyc, a) + math.radians(0.8) * wave(f, cyc * 2, b),
            math.radians(1.5) * wave(f, cyc, c),
            math.radians(2.0) * wave(f, cyc, b + 0.2)))()
    for f in range(1, LOOP + 1):
        for n, fn in keys.items():
            pb = arm.pose.bones[n]
            pb.rotation_euler = fn(f)
            pb.keyframe_insert("rotation_euler", frame=f)
    # Ciclo explícito para que el último fotograma enlace con el primero.
    # (los valores de f=1 y f=LOOP+1 coinciden; el clip termina en LOOP y vuelve a 1)
    for pb in arm.pose.bones:
        pb.rotation_euler = (0, 0, 0)

    # ---- Comprobaciones ----
    assert len(obj.data.vertices) == nv0 and len(obj.data.polygons) == nf0, "la malla ha cambiado"
    unweighted = sum(1 for v in obj.data.vertices if sum(g.weight for g in v.groups) < 0.99)
    sc.frame_set(1)
    dg = bpy.context.evaluated_depsgraph_get()
    rest = np.array([v.co[:] for v in obj.data.vertices])
    maxd = 0.0
    trunk_move = 0.0
    ev_idx = np.array([e.vertices[:] for e in obj.data.edges])
    len0 = np.linalg.norm(rest[ev_idx[:, 0]] - rest[ev_idx[:, 1]], axis=1)
    stretch = 0.0
    static = mask < 0.001
    for f in range(1, LOOP + 1, 6):
        sc.frame_set(f)
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        cur = np.array([v.co[:] for v in ev.data.vertices])
        dd = np.linalg.norm(cur - rest, axis=1)
        maxd = max(maxd, float(dd.max()))
        trunk_move = max(trunk_move, float(dd[static].max()) if static.any() else 0.0)
        ln = np.linalg.norm(cur[ev_idx[:, 0]] - cur[ev_idx[:, 1]], axis=1)
        ok = len0 > 1e-5
        stretch = max(stretch, float(np.abs(ln[ok] / len0[ok] - 1).max()))
    sc.frame_set(1)
    print(f"MANZANO {name}: verts={nv0} caras={nf0} copa desde z={z0:.2f} (soldada={fused}) ramas={k} "
          f"piezas_rigidas={len(rigid)} sin_peso={unweighted} mov_max={maxd * 100:.1f}cm tronco_mov={trunk_move * 1000:.2f}mm "
          f"verts_copa={int(foliage.sum())}/{len(co)} estiramiento_max_arista={stretch * 100:.1f}%")

    if "pesos" in ARGS:
        render_weights(obj, mask, name)

    blend = os.path.join(DIR, name + "_anim.blend")
    fbx = os.path.join(DIR, name + "_anim.fbx")
    for o in bpy.data.objects:
        o.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    if os.path.exists(blend + "1"):
        os.remove(blend + "1")
    bpy.ops.export_scene.fbx(
        filepath=fbx, object_types={"ARMATURE", "MESH"}, use_selection=False,
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE",
    )
    print("RIG_OK", blend, fbx)


def render_weights(obj, mask, name):
    out = os.path.join(DIR, "_pesos")
    os.makedirs(out, exist_ok=True)
    attr = obj.data.color_attributes.new("peso", 'FLOAT_COLOR', 'POINT')
    for i, m in enumerate(mask):
        attr.data[i].color = (0.45 + 0.4 * (1 - m), 0.3 + 0.5 * m, 0.2, 1)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = sc.render.resolution_y = 600
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 1.35
    for i, ang in enumerate([0, 90]):
        a = math.radians(ang)
        cam.location = Vector((math.sin(a) * 4, -math.cos(a) * 4, 0.55))
        cam.rotation_euler = (math.radians(90), 0, a)
        sc.render.filepath = os.path.join(out, f"{name}_{i}.png")
        bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)
    obj.data.color_attributes.remove(attr)


paths = [p for p in sorted(glob.glob(os.path.join(DIR, "*.fbx"))) if not p.endswith("_anim.fbx")]
for i, p in enumerate(paths):
    build(p, i)
