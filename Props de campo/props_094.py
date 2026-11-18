"""Props de campo nuevos (0.9.4): rig + animación + LODs con Blender CLI.

blender -b --factory-startup -P props_094.py -- [grupo ...] [vistas=dir]
  grupos: arbustos maizales maizdanado espantapajaros tractor carretilla nido   (sin nada = todos)

Sólo modelos CON textura (los "Props del granero" aún no tienen y se quedan fuera).
Salida por grupo en <grupo>/Animados/<Nombre>_anim.blend/.fbx (estáticos: <Nombre>_LOD.blend/.fbx). Los FBX originales
no se tocan.

Cómo se evita romper la malla: los pesos NO se pintan por pieza; salen de un campo continuo de la posición (altura
normalizada y, en el espantapájaros, regiones con bordes suaves). Dos puntos que se tocan tienen casi el mismo peso, así
que hojas, pétalos y pajas que se solapan se mueven juntos y no se separan ni se abren grietas. Los ángulos son pequeños
(viento suave) y el script mide en cada clip el estiramiento de las aristas (p99.9 y máximo) y lo imprime: si pasa de
los topes, falla. Los LODs (Decimate colapso) reciben los pesos con la misma función, así se mueven igual.

Clips (30 fps):
  arbustos         Idle_Viento (4 s, bucle)  Sacudida (0,9 s: una gallina se mete o lo roza)
  maizales         Idle_Viento (3,6 s, bucle) Sacudida (0,9 s)
  maizdanado       Idle_Viento (5 s, bucle, mustio)
  espantapajaros   Idle_Viento (5 s, bucle)  Sacudida (1,2 s: picotazo)   (también el roto, más caído)
  tractor          Motor (1 s, bucle: vibración del chasis). Ruedas = pivotes rígidos que gira el juego.
  carretilla, nido estáticos (sólo LODs).
"""
import bpy, bmesh, sys, os, math, glob
import numpy as np
from mathutils import Vector, Matrix, Euler

BASE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SOLO = next((set(a.split("=", 1)[1].split(",")) for a in argv if a.startswith("solo=")), None)
VISTAS = next((a.split("=", 1)[1] for a in argv if a.startswith("vistas=")), None)
GRUPOS = [a for a in argv if "=" not in a and a not in ("sinparar", "diag")] or ["arbustos", "maizales", "maizdanado", "espantapajaros", "tractor", "carretilla", "nido"]
FPS = 30
TOPE_P999, TOPE_MAX = 0.06, 0.15   # estiramiento de aristas permitido (6 % casi todas, 15 % la peor)
RESUMEN = []


def P(*a):
    return os.path.join(BASE, *a)


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ importar

def nueva_escena():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS
    return sc


def importar(fbx, png, nombre):
    antes = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=fbx)
    nuevos = [o for o in bpy.data.objects if o not in antes]
    mallas = [o for o in nuevos if o.type == "MESH"]
    assert mallas, f"{fbx}: sin malla"
    src = mallas[0]
    if len(mallas) > 1:  # varias mallas: se juntan (comparten textura)
        bpy.ops.object.select_all(action="DESELECT")
        for m in mallas:
            m.select_set(True)
        bpy.context.view_layer.objects.active = src
        bpy.ops.object.join()
    for o in nuevos:
        if o.name in bpy.data.objects and o != src:
            bpy.data.objects.remove(o)
    assert src.data.uv_layers, f"{nombre} no tiene UV"
    src.data.transform(src.matrix_world)
    src.matrix_world = Matrix.Identity(4)
    co = np.array([v.co[:] for v in src.data.vertices])
    lo, hi = co.min(0), co.max(0)
    src.data.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    src.name = src.data.name = nombre
    mat = bpy.data.materials.new(nombre)
    mat.use_nodes = True
    nt = mat.node_tree
    ti = nt.nodes.new("ShaderNodeTexImage")
    ti.image = bpy.data.images.load(png, check_existing=True)
    nt.links.new(ti.outputs["Color"], next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"])
    src.data.materials.clear()
    src.data.materials.append(mat)
    for p in src.data.polygons:
        p.material_index = 0
    return src


def coords(o):
    return np.array([v.co[:] for v in o.data.vertices])


def lod(src, nombre, objetivo):
    o = src.copy()
    o.data = src.data.copy()
    o.name = o.data.name = nombre
    bpy.context.scene.collection.objects.link(o)
    for m in list(o.modifiers):
        o.modifiers.remove(m)
    o.vertex_groups.clear()
    # Vértices duplicados (piezas cosidas con vértices repetidos) bloquean el colapso y lo dejan en astillas (arbusto 18):
    # se fusionan antes (a 0,3 mm; los UV van por esquina, no se pierden).
    # Si la fusión se come geometría (hojas de dos caras muy finas, arbusto 18), no se aplica.
    t0 = tris(o)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0003)
    bm.faces.ensure_lookup_table()
    if sum(len(f.verts) - 2 for f in bm.faces) >= t0 * 0.97:
        bm.to_mesh(o.data)
    bm.free()
    t = tris(o)
    if objetivo < t:
        dm = o.modifiers.new("Reducir", "DECIMATE")
        dm.decimate_type = "COLLAPSE"
        dm.ratio = objetivo / t
        dm.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=dm.name)
    return o


# ------------------------------------------------------------------ esqueleto y pesos

def esqueleto(nombre, huesos):
    """huesos: [(nombre, padre, cabeza, cola)] en metros de la malla."""
    arm_d = bpy.data.armatures.new(nombre + "_Rig")
    arm = bpy.data.objects.new(nombre + "_Rig", arm_d)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for n, padre, cab, cola in huesos:
        b = arm_d.edit_bones.new(n)
        b.head, b.tail = Vector(cab), Vector(cola)
        if padre:
            b.parent = arm_d.edit_bones[padre]
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
    return arm


def pesar(o, arm, campos):
    """campos: {hueso: array de pesos por vértice}; se normalizan a suma 1."""
    total = sum(campos.values())
    total = np.where(total < 1e-6, 1.0, total)
    o.vertex_groups.clear()
    for n, w in campos.items():
        w = w / total
        g = o.vertex_groups.new(name=n)
        for i, x in enumerate(w):
            if x > 1e-4:
                g.add([i], float(x), "REPLACE")
    m = o.modifiers.new("Armature", "ARMATURE")
    m.object = arm
    o.parent = arm


def cadena(t, nudos):
    """Pesos en cadena por altura normalizada t: un 'sombrero' suave por nudo (suman 1)."""
    k = len(nudos)
    w = []
    for i in range(k):
        a = smooth(nudos[i - 1], nudos[i], t) if i > 0 else np.ones_like(t)
        b = 1 - smooth(nudos[i], nudos[i + 1], t) if i < k - 1 else np.ones_like(t)
        w.append(a * b)
    return w


# ------------------------------------------------------------------ clips

def clip(arm, nombre, segundos, fn, bucle=True):
    """fn(t, T) -> {hueso: (rx, ry, rz) grados}  o  {hueso: ((rx,ry,rz), (tx,ty,tz))}."""
    act = bpy.data.actions.new(nombre)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    n = round(segundos * FPS)
    for f in range(n + 1):
        t = (0.0 if (bucle and f == n) else f / FPS)
        pose = fn(t, segundos)
        for pb in arm.pose.bones:
            v = pose.get(pb.name, (0, 0, 0))
            rot, loc = (v if isinstance(v[0], tuple) else (v, (0, 0, 0)))
            pb.rotation_euler = Euler(tuple(math.radians(x) for x in rot), "XYZ")
            pb.location = Vector(loc)
            pb.keyframe_insert("rotation_euler", frame=f + 1)
            pb.keyframe_insert("location", frame=f + 1)
    act.frame_range = (1, n + 1)
    return act


def estiramiento(o, arm, act):
    """Cambio relativo de longitud de las aristas en todos los fotogramas del clip (p99.9, máximo)."""
    arm.animation_data.action = act
    me = o.data
    ed = np.array([e.vertices[:] for e in me.edges])
    rest = coords(o)
    l0 = np.linalg.norm(rest[ed[:, 0]] - rest[ed[:, 1]], axis=1)
    ok = l0 > 1e-6
    peor = []
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    dg = bpy.context.evaluated_depsgraph_get()
    for f in range(f0, f1 + 1, 2):
        bpy.context.scene.frame_set(f)
        dg.update()
        ev = o.evaluated_get(dg)
        c = np.array([v.co[:] for v in ev.data.vertices])
        l = np.linalg.norm(c[ed[:, 0]] - c[ed[:, 1]], axis=1)
        peor.append(np.abs(l[ok] - l0[ok]) / l0[ok])
    if "diag" in argv:
        allp = np.stack(peor)                     # (fotogramas, aristas ok)
        e_max = allp.max(0)
        idx = np.argsort(e_max)[-12:]
        mid = (rest[ed[ok][:, 0]] + rest[ed[ok][:, 1]]) / 2
        H = rest[:, 2].max(); W = rest[:, 0].max() - rest[:, 0].min()
        for i in idx[::-1]:
            print(f"   DIAG {act.name}: {e_max[i] * 100:.1f}% en t={mid[i, 2] / H:.2f} x={mid[i, 0] / W:+.2f} y={mid[i, 1] / W:+.2f} l0={l0[ok][i] * 1000:.2f}mm")
    peor = np.concatenate(peor)
    bpy.context.scene.frame_set(1)
    return float(np.percentile(peor, 99.9)), float(peor.max())


def revisar(nombre, o, arm, acts):
    for a in acts:
        p, mx = estiramiento(o, arm, a)
        estado = "OK" if (p <= TOPE_P999 and mx <= TOPE_MAX) else "FALLA"
        RESUMEN.append(f"{nombre}/{a.name}: estiramiento p99.9={p * 100:.2f}% max={mx * 100:.2f}% {estado}")
        print(RESUMEN[-1])
        if estado == "FALLA" and "sinparar" not in argv:
            raise SystemExit(f"{nombre}/{a.name}: la malla se estira demasiado")


# ------------------------------------------------------------------ exportar y vistas

def exportar(objs, fbx, anim=True, aplicar=False):
    blend = fbx[:-4] + ".blend"
    os.makedirs(os.path.dirname(fbx), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    if os.path.exists(blend + "1"):
        os.remove(blend + "1")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.ops.export_scene.fbx(
        filepath=fbx, object_types={"ARMATURE", "MESH", "EMPTY"}, use_selection=True,
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
        bake_anim=anim, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE", bake_space_transform=aplicar)
    print("EXPORTADO", fbx)


def vista(nombre, objs, arm=None, act=None, frame=None, extra=""):
    """Render con textura: los objetos en fila (LOD0, LOD1, LOD2), opcionalmente en un fotograma del clip."""
    if not VISTAS:
        return
    sc = bpy.context.scene
    if arm is not None and act is not None:
        arm.animation_data.action = act
        sc.frame_set(frame or 1)
    for o in sc.objects:
        o.hide_render = o.type == "MESH" and o not in objs
    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        ev = o.evaluated_get(dg)
        for v in ev.data.vertices:
            w = o.matrix_world @ v.co
            mn = Vector(map(min, mn, w)); mx = Vector(map(max, mx, w))
    size = mx - mn
    paso = max(size.x, size.y) * 1.15 + 0.05
    guard = {}
    for i, o in enumerate(objs):
        tgt = o.parent if o.parent is not None and o.parent.type == "ARMATURE" and len(objs) == 1 else o
        guard[o] = o.matrix_world.copy()
    # en fila sobre X (si cuelgan de un esqueleto se mueve una copia evaluada)
    shown = []
    for i, o in enumerate(objs):
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        c = bpy.data.objects.new(o.name + "_vista", me)
        sc.collection.objects.link(c)
        c.matrix_world = Matrix.Translation((i * paso, 0, 0)) @ o.matrix_world
        shown.append(c)
    for o in sc.objects:
        if o.type == "MESH":
            o.hide_render = o not in shown
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x, sc.render.resolution_y = 520 * len(shown), 560
    cam = bpy.data.objects.new("VistaCam", bpy.data.cameras.new("VistaCam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(paso * len(shown), size.z * len(shown) * 0.95) * 1.02
    d = Vector((0.35, -1, 0.35)).normalized()
    tgt = Vector(((len(shown) - 1) * paso / 2 + (mn.x + mx.x) / 2, (mn.y + mx.y) / 2, (mn.z + mx.z) / 2))
    cam.location = tgt + d * 12
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    os.makedirs(VISTAS, exist_ok=True)
    sc.render.filepath = os.path.join(VISTAS, f"{nombre}{extra}.png")
    bpy.ops.render.render(write_still=True)
    for c in shown:
        bpy.data.objects.remove(c)
    bpy.data.objects.remove(cam)
    for o in sc.objects:
        o.hide_render = False


# ------------------------------------------------------------------ plantas que se mecen

def planta(nombre, fbx, png, salida, pivotes, bandas, clips, objetivos):
    """Arbustos y maíz: cadena de huesos Raiz -> Tallo_1 -> Tallo_2...; los pesos dependen sólo de la altura normalizada.
    Cadena ANIDADA con bandas anchas: s_i = smoothstep(banda_i) es la parte que mueve el Tallo_i y los que cuelgan de
    él; w_Raiz = 1 - s_1, w_Tallo_i = s_i - s_(i+1). Las bandas abarcan casi toda la altura (los arbustos son más anchos
    que altos: un cambio brusco de peso abajo cizallaba sus bordes un 20-30 %); la base (tierra) queda fija."""
    nueva_escena()
    src = importar(fbx, png, nombre)
    co = coords(src)
    H = float(co[:, 2].max())
    k = len(pivotes)
    nombres = ["Raiz"] + [f"Tallo_{i}" for i in range(1, k + 1)]
    huesos = [("Raiz", None, (0, 0, 0), (0, 0, max(pivotes[0] * H, 0.02)))]
    for i in range(k):
        z0 = pivotes[i] * H
        z1 = pivotes[i + 1] * H if i + 1 < k else H * 1.02
        huesos.append((nombres[i + 1], nombres[i], (0, 0, z0), (0, 0, max(z1, z0 + 0.02))))
    arm = esqueleto(nombre, huesos)

    def campos(o):
        t = coords(o)[:, 2] / H
        sv = [smooth(a_, b_, t) for a_, b_ in bandas] + [np.zeros_like(t)]
        w = [1 - sv[0]] + [np.clip(sv[i] - sv[i + 1], 0, 1) for i in range(k)]
        return dict(zip(nombres, w))

    lods = []
    for i, obj in enumerate(objetivos):
        o = lod(src, f"{nombre}_LOD{i}", obj)
        pesar(o, arm, campos(o))
        lods.append(o)
    src.hide_render = True
    src.hide_set(True)
    # Las plantas planas (mucho más anchas que altas) giran menos: el borde lejano se desplaza más por grado.
    Wd = float(max(co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min()))
    kk = float(np.clip(1.6 * H / max(Wd, 1e-3), 0.4, 1.0))

    def escalar(fn):
        def g(t, T):
            return {b: tuple(x * kk for x in r) for b, r in fn(t, T).items()}
        return g
    acts = [clip(arm, cn, T, escalar(fn), bucle) for cn, T, fn, bucle in clips]
    revisar(nombre, lods[0], arm, acts)
    RESUMEN.append(f"{nombre}: amplitud x{kk:.2f}, alto {H:.3f} m (sin escala), tris original {tris(src)}, LODs {[tris(o) for o in lods]}")
    arm.animation_data.action = acts[0]
    exportar([arm] + lods, salida)
    vista(nombre, lods)
    pico = acts[0]
    vista(nombre, [lods[0]], arm, pico, frame=int(pico.frame_range[1] * 0.25), extra="_pose")
    return H


def ondas(amp, fases, T, t, armon=0.35):
    w = 2 * math.pi / T
    return amp * (math.sin(w * t + fases) + armon * math.sin(2 * w * t + fases * 1.7))


def viento_arbusto(t, T):
    return {"Tallo_1": (ondas(1.1, 0.0, T, t), 0, ondas(0.8, 1.1, T, t)),
            "Tallo_2": (ondas(1.6, -0.7, T, t), 0, ondas(1.1, 0.4, T, t))}


def sacudida(amp1, amp2, f=5.0, d=4.5):
    def fn(t, T):
        e = math.exp(-d * t) * math.sin(2 * math.pi * f * t) * (1 - t / T)
        e2 = math.exp(-d * t) * math.sin(2 * math.pi * f * t + 0.9) * (1 - t / T)
        return {"Tallo_1": (amp1 * e, 0, amp1 * 0.6 * e2), "Tallo_2": (amp2 * e2, 0, amp2 * 0.7 * e)}
    return fn


def viento_maiz(escala=1.0):
    def fn(t, T):
        return {"Tallo_1": (ondas(1.0 * escala, 0.0, T, t), 0, ondas(0.7 * escala, 1.3, T, t)),
                "Tallo_2": (ondas(1.8 * escala, -0.6, T, t), 0, ondas(1.2 * escala, 0.7, T, t)),
                "Tallo_3": (ondas(2.6 * escala, -1.2, T, t), 0, ondas(1.8 * escala, 0.1, T, t))}
    return fn


def sacudida_maiz(t, T):
    e = math.exp(-4.0 * t) * math.sin(2 * math.pi * 4.5 * t) * (1 - t / T)
    return {"Tallo_1": (2.0 * e, 0, 1.2 * e), "Tallo_2": (3.5 * e, 0, 2.0 * e), "Tallo_3": (5.0 * e, 0, 3.0 * e)}


# Arbusto 18: hojas de lámina fina con dos caras; por debajo de ~6k triángulos el colapso las deja en astillas.
OBJETIVO_ARBUSTO = {"18": [12000, 8000, 6000]}


def g_arbustos():
    for f in sorted(glob.glob(P("Arbustos", "Arbustos", "Arbusto *.fbx")), key=lambda s: int("".join(c for c in os.path.basename(s) if c.isdigit()))):
        n = "".join(c for c in os.path.basename(f) if c.isdigit())
        if SOLO and n not in SOLO: continue
        png = P("Arbustos", "Textura", f"Arbusto {n}.png")
        if not os.path.exists(png):
            print("SIN TEXTURA, fuera:", f); continue
        planta(f"Arbusto{n}", f, png, P("Arbustos", "Animados", f"Arbusto{n}_anim.fbx"), [0.0, 0.45], [(-0.1, 0.9), (0.3, 1.25)],
               [("Idle_Viento", 4.0, viento_arbusto, True), ("Sacudida", 0.9, sacudida(3.0, 4.5), False)],
               OBJETIVO_ARBUSTO.get(n, [4000, 2000, 1100]))


def g_maizales():
    for f in sorted(glob.glob(P("Maizales", "Finales", "Maizal *.fbx"))):
        n = "".join(c for c in os.path.basename(f) if c.isdigit())
        if SOLO and n not in SOLO: continue
        png = P("Maizales", "Texturas", f"Maizal {n}.png")
        if not os.path.exists(png):
            print("SIN TEXTURA, fuera:", f); continue
        planta(f"Maizal{n}", f, png, P("Maizales", "Animados", f"Maizal{n}_anim.fbx"), [0.05, 0.35, 0.65], [(0.04, 0.6), (0.25, 0.95), (0.5, 1.25)],
               [("Idle_Viento", 3.6, viento_maiz(), True), ("Sacudida", 0.9, sacudida_maiz, False)],
               [4000, 1800, 900])


def g_maizdanado():
    for f in sorted(glob.glob(P("Maiz dañado", "Finales", "*.fbx"))):
        n = "".join(c for c in os.path.basename(f) if c.isdigit())
        png = glob.glob(P("Maiz dañado", "Texturas", f"Maiz_{n}*.png"))
        if not png:
            print("SIN TEXTURA, fuera:", f); continue
        planta(f"MaizRoto{n}", f, png[0], P("Maiz dañado", "Animados", f"MaizRoto{n}_anim.fbx"), [0.12, 0.4, 0.7], [(0.12, 0.7), (0.35, 1.0), (0.6, 1.3)],
               [("Idle_Viento", 5.0, viento_maiz(0.6), True)], [3000, 1200, 450])


# ------------------------------------------------------------------ espantapájaros

def marcas_espantapajaros(co):
    """Dónde cortar los huesos, por el perfil de anchura (60 franjas de altura). Cada corte es (centro, media banda) en
    altura normalizada y cae donde la malla es ESTRECHA o no hay nada: el giro relativo entre dos huesos cizalla lo que
    queda dentro de la banda en proporción a su anchura, así que en el palo o en un hueco no se nota nada.
      cintura: hueco sin vértices entre palo y cuerpo (el roto) o, si no hay, el tramo estrecho del palo.
      cuello:  justo encima de los brazos (el pañuelo), banda ancha.
      ala:     donde el sombrero se ensancha por encima de la cara."""
    H = co[:, 2].max()
    W = co[:, 0].max() - co[:, 0].min()
    t = co[:, 2] / H
    bins = 60
    ancho = np.zeros(bins)
    for b in range(bins):
        m = (t >= b / bins) & (t < (b + 1) / bins)
        if m.sum() > 5:
            ancho[b] = (np.percentile(co[m, 0], 99) - np.percentile(co[m, 0], 1)) / W
    brazos = np.where(ancho > 0.75)[0]
    b_lo, b_hi = (brazos.min(), brazos.max()) if len(brazos) else (int(bins * 0.5), int(bins * 0.6))

    def tramos(cond):
        out, ini = [], None
        for b in range(b_lo):
            if cond(b) and ini is None: ini = b
            if not cond(b) and ini is not None: out.append((ini, b)); ini = None
        if ini is not None: out.append((ini, b_lo))
        return out
    huecos = [r for r in tramos(lambda b: ancho[b] == 0.0) if r[0] > 3]
    palos = [r for r in tramos(lambda b: 0.0 < ancho[b] < 0.16) if r[0] > 3]
    if huecos:
        r = max(huecos, key=lambda r: r[1] - r[0]); tipo = "hueco"
    elif palos:
        r = max(palos, key=lambda r: r[1] - r[0]); tipo = "palo"
    else:
        r = (int(bins * 0.3), int(bins * 0.36)); tipo = "?"
    cintura = ((r[0] + r[1]) / 2 / bins, max((r[1] - r[0]) / 2 / bins, 0.012), tipo)
    cuello = ((b_hi + 1) / bins + 0.02, 0.06)
    ref = ancho[min(bins - 1, b_hi + 3): min(bins, b_hi + 6)]
    ref = ref[ref > 0].mean() if (ref > 0).any() else 0.4
    ala = 0.86
    for b in range(int((cuello[0] + 0.07) * bins), bins):
        if ancho[b] > ref * 1.12:
            ala = b / bins; break
    return dict(H=H, W=W, cintura=cintura, brazos=(b_lo / bins, (b_hi + 1) / bins), cuello=cuello, ala=(ala, 0.05), perfil=ancho)


def espantapajaros(nombre, fbx, png, salida, roto=False):
    nueva_escena()
    src = importar(fbx, png, nombre)
    co = coords(src)
    mk = marcas_espantapajaros(co)
    H, W = mk["H"], mk["W"]
    print("PERFIL", " ".join(f"{x:.2f}" for x in mk["perfil"]))
    txt = (f"cintura {mk['cintura'][0]:.2f}±{mk['cintura'][1]:.2f} ({mk['cintura'][2]}) brazos {mk['brazos'][0]:.2f}-"
           f"{mk['brazos'][1]:.2f} cuello {mk['cuello'][0]:.2f} ala {mk['ala'][0]:.2f}")
    print(f"MARCAS {nombre}: {txt}")
    RESUMEN.append(f"{nombre}: marcas {txt}")
    (tc, hc, _), (tn, hn), (ta, ha) = mk["cintura"], mk["cuello"], mk["ala"]
    zc, zn, za = tc * H, tn * H, ta * H
    zb = (mk["brazos"][0] + mk["brazos"][1]) / 2 * H
    xm = W * 0.36  # donde acaban las mangas y empiezan las manos de paja
    huesos = [("Raiz", None, (0, 0, 0), (0, 0, zc)),
              ("Cuerpo", "Raiz", (0, 0, zc), (0, 0, zn)),
              ("Cabeza", "Cuerpo", (0, 0, zn), (0, 0, za)),
              ("Sombrero", "Cabeza", (0, 0, za), (0, 0, H)),
              ("Mano.L", "Cuerpo", (xm, 0, zb), (W / 2, 0, zb)),
              ("Mano.R", "Cuerpo", (-xm, 0, zb), (-W / 2, 0, zb))]
    arm = esqueleto(nombre, huesos)

    def campos(o):
        c = coords(o)
        z, x = c[:, 2] / H, c[:, 0]
        cuerpo = smooth(tc - hc, tc + hc, z)
        cabeza = smooth(tn - hn, tn + hn, z)
        somb = smooth(ta - ha, ta + ha, z)
        # Manos: sólo por la distancia al centro (más allá de las mangas sólo hay manos de paja); el corte vertical va
        # lejos de las pajas (antes atravesaba las de arriba y las estiraba un 22 %).
        b0, b1 = mk["brazos"]
        en_brazos = smooth(b0 - 0.14, b0 - 0.06, z) * (1 - smooth(b1 + 0.04, b1 + 0.10, z))
        w_ml = smooth(xm - 0.07 * W, xm + 0.07 * W, x) * en_brazos
        w_mr = smooth(xm - 0.07 * W, xm + 0.07 * W, -x) * en_brazos
        resto = 1 - w_ml - w_mr
        w_s = somb * resto
        w_c = cabeza * (1 - somb) * resto
        resto2 = resto - w_s - w_c
        w_b = cuerpo * resto2
        w_r = resto2 - w_b
        return {"Raiz": w_r, "Cuerpo": w_b, "Cabeza": w_c, "Sombrero": w_s, "Mano.L": w_ml, "Mano.R": w_mr}

    lods = []
    for i, obj in enumerate([8000, 3000, 1200]):
        o = lod(src, f"{nombre}_LOD{i}", obj)
        pesar(o, arm, campos(o))
        lods.append(o)
    src.hide_set(True)
    k = 1.6 if roto else 1.0   # el roto cuelga suelto del palo partido: se balancea más (su corte cae en un hueco)

    def viento(t, T):
        return {"Cuerpo": (ondas(1.0 * k, 0.0, T, t), 0, ondas(1.6 * k, 0.9, T, t)),
                "Cabeza": (ondas(0.8, 0.6, T, t), 0, ondas(1.2, 1.5, T, t)),
                "Sombrero": (ondas(0.7, 1.2, T, t), 0, ondas(1.0, 2.2, T, t)),
                "Mano.L": (ondas(4.0, 0.3, T / 2, t, 0.2), 0, ondas(1.5, 1.0, T, t)),
                "Mano.R": (ondas(4.0, 2.1, T / 2, t, 0.2), 0, ondas(1.5, 2.6, T, t))}

    def golpe(t, T):
        e = math.exp(-3.5 * t) * math.sin(2 * math.pi * 3.2 * t) * (1 - t / T)
        e2 = math.exp(-3.5 * t) * math.sin(2 * math.pi * 3.2 * t + 1.0) * (1 - t / T)
        return {"Cuerpo": (3.0 * e, 0, 5.0 * e2), "Cabeza": (1.2 * e2, 0, 1.8 * e),
                "Sombrero": (1.0 * e, 0, 1.5 * e2), "Mano.L": (6.0 * e2, 0, 0), "Mano.R": (-6.0 * e2, 0, 0)}

    acts = [clip(arm, "Idle_Viento", 5.0, viento), clip(arm, "Sacudida", 1.2, golpe, False)]
    revisar(nombre, lods[0], arm, acts)
    RESUMEN.append(f"{nombre}: alto {H:.3f}, tris original {tris(src)}, LODs {[tris(o) for o in lods]}")
    arm.animation_data.action = acts[0]
    exportar([arm] + lods, salida)
    vista(nombre, lods)
    vista(nombre, [lods[0]], arm, acts[1], frame=6, extra="_golpe")
    if VISTAS:  # pesos: color por hueso dominante
        pinta_pesos(nombre, lods[0])


def pinta_pesos(nombre, o):
    cols = [(0.6, 0.6, 0.6), (0.9, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.3, 0.95), (0.95, 0.85, 0.1), (0.9, 0.3, 0.9)]
    me = bpy.data.meshes.new_from_object(o)
    c = bpy.data.objects.new(nombre + "_pesos", me)
    bpy.context.scene.collection.objects.link(c)
    vc = me.color_attributes.new("w", "FLOAT_COLOR", "POINT")
    gi = {g.index: k for k, g in enumerate(o.vertex_groups)}
    for v in o.data.vertices:
        best = max(v.groups, key=lambda g: g.weight) if v.groups else None
        mix = [0.0, 0.0, 0.0]
        for g in v.groups:
            col = cols[gi[g.group] % len(cols)]
            for j in range(3):
                mix[j] += col[j] * g.weight
        vc.data[v.index].color = (*mix, 1)
    sc = bpy.context.scene
    sc.display.shading.color_type = "VERTEX"
    for ob in sc.objects:
        ob.hide_render = ob.type == "MESH" and ob != c
    cam = bpy.data.objects.new("VC", bpy.data.cameras.new("VC")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"; cam.data.ortho_scale = max(o.dimensions) * 1.1
    cam.location = (0, -10, o.dimensions.z / 2); cam.rotation_euler = (math.radians(90), 0, 0)
    sc.render.resolution_x = sc.render.resolution_y = 700
    sc.render.filepath = os.path.join(VISTAS, f"{nombre}_pesos.png")
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(c); bpy.data.objects.remove(cam)
    sc.display.shading.color_type = "TEXTURE"


def g_espantapajaros():
    espantapajaros("Espantapajaros", P("Maiz dañado", "Espantapajaros_final.fbx"), P("Espanta pajaros", "Espantapajaros_color.png"),
                   P("Espanta pajaros", "Animados", "Espantapajaros_anim.fbx"))
    espantapajaros("EspantapajarosRoto", P("Espanta Pajaros dañado", "Roto_espantapajaros_final.fbx"),
                   P("Espanta Pajaros dañado", "RotoEspantapajaros_color.png"),
                   P("Espanta Pajaros dañado", "Animados", "EspantapajarosRoto_anim.fbx"), roto=True)


# ------------------------------------------------------------------ tractor (pivotes rígidos)

# Medidas sobre Tractor_final normalizado (largo 1,0 en Y, frente hacia -Y, base en z=0), vistas con rejilla:
# rueda trasera r 0,18 en y +0,25, x ±0,242; delantera r 0,115 en y -0,29, x ±0,21 (por fuera de la carrocería).
RUEDAS = [("Rueda_TI", (-0.242, 0.25, 0.18), 0.18), ("Rueda_TD", (0.242, 0.25, 0.18), 0.18),
          ("Rueda_DI", (-0.21, -0.29, 0.115), 0.115), ("Rueda_DD", (0.21, -0.29, 0.115), 0.115)]


def g_tractor():
    nueva_escena()
    body = importar(P("Tractor", "Tractor_final.fbx"), P("Tractor", "Tractor-color.png"), "Tractor")
    rueda = importar(P("Tractor", "Llanta_tractor_final.fbx"), P("Tractor", "LlantaTractor-color.png"), "Llanta")
    # La llanta: eje sobre Y en el original -> centrada en su eje, diámetro 1.
    co = coords(rueda)
    lo, hi = co.min(0), co.max(0)
    rueda.data.transform(Matrix.Translation((0, 0, -(lo[2] + hi[2]) / 2)))
    diam = float(max(hi[0] - lo[0], hi[2] - lo[2]))
    rueda.data.transform(Matrix.Scale(1.0 / diam, 4))
    rueda.data.transform(Matrix.Rotation(math.radians(90), 4, "Z"))  # eje -> X (el tractor va sobre Y)
    raiz = bpy.data.objects.new("Tractor_Raiz", None); bpy.context.scene.collection.objects.link(raiz)
    chasis = bpy.data.objects.new("Chasis", None); bpy.context.scene.collection.objects.link(chasis)
    chasis.parent = raiz
    objs = [raiz, chasis]
    for lvl, (ob_c, ob_r) in enumerate([(8000, 2500), (3000, 1000), (1200, 400)]):
        c = lod(body, f"Tractor_LOD{lvl}", ob_c)
        c.parent = chasis
        objs.append(c)
        for n, pos, r in RUEDAS:
            piv = bpy.data.objects.get(n)
            if piv is None:
                piv = bpy.data.objects.new(n, None); bpy.context.scene.collection.objects.link(piv)
                piv.parent = raiz
                piv.location = pos
                objs.append(piv)
            w = lod(rueda, f"{n}_LOD{lvl}", ob_r)
            w.data.transform(Matrix.Scale(2 * r, 4))
            if n.endswith("D"):  # las de la derecha, con la cara buena hacia fuera
                w.data.transform(Matrix.Rotation(math.radians(180), 4, "Z"))
            w.parent = piv
            objs.append(w)
    body.hide_set(True); rueda.hide_set(True)
    body.hide_render = rueda.hide_render = True
    # Motor: el chasis vibra (los pivotes de las ruedas no se animan: los gira el juego).
    act = bpy.data.actions.new("Motor"); act.use_fake_user = True
    chasis.animation_data_create(); chasis.animation_data.action = act
    n = FPS
    for f in range(n + 1):
        t = 0.0 if f == n else f / FPS
        w = 2 * math.pi
        chasis.location = (0, 0.002 * math.sin(w * 3 * t), 0.004 * abs(math.sin(w * 6 * t)))
        chasis.rotation_euler = (math.radians(0.35 * math.sin(w * 6 * t + 0.5)), math.radians(0.5 * math.sin(w * 3 * t)), 0)
        chasis.keyframe_insert("location", frame=f + 1)
        chasis.keyframe_insert("rotation_euler", frame=f + 1)
    RESUMEN.append(f"Tractor: chasis tris {tris(body)} -> {[tris(bpy.data.objects[f'Tractor_LOD{i}']) for i in range(3)]}, "
                   f"rueda {tris(rueda)} -> {[tris(bpy.data.objects[f'Rueda_TI_LOD{i}']) for i in range(3)]}")
    # Jerarquía de vacíos (sin esqueleto): con la conversión de ejes sólo en la raíz, Unity lo dejaba de pie. Se aplica
    # la transformación a los datos (bake_space_transform) para que llegue derecho.
    exportar(objs, P("Tractor", "Animados", "Tractor_anim.fbx"), aplicar=True)
    if VISTAS:
        sc = bpy.context.scene
        chasis.animation_data.action = None
        vista("Tractor", [o for o in objs if o.type == "MESH" and o.name.endswith("_LOD0")][:1])
        # montado: todas las piezas LOD0 juntas
        lod0 = [o for o in objs if o.type == "MESH" and o.name.endswith("_LOD0")]
        me = []
        dg = bpy.context.evaluated_depsgraph_get()
        tmp = []
        for o in lod0:
            m = bpy.data.meshes.new_from_object(o.evaluated_get(dg)); m.transform(o.matrix_world)
            x = bpy.data.objects.new(o.name + "_m", m); bpy.context.scene.collection.objects.link(x)
            x.data.materials.clear(); x.data.materials.append(o.data.materials[0]); tmp.append(x)
        bpy.ops.object.select_all(action="DESELECT")
        for x in tmp: x.select_set(True)
        bpy.context.view_layer.objects.active = tmp[0]
        bpy.ops.object.join()
        junto = bpy.context.view_layer.objects.active
        vista("Tractor_montado", [junto])
        for ang, nm in ((90, "_lado"), (0, "_frente")):
            junto.rotation_euler.z = math.radians(ang)
            vista("Tractor_montado", [junto], extra=nm)
        bpy.data.objects.remove(junto)


# ------------------------------------------------------------------ estáticos

def estatico(nombre, fbx, png, salida, objetivos):
    nueva_escena()
    src = importar(fbx, png, nombre)
    lods = [lod(src, f"{nombre}_LOD{i}", obj) for i, obj in enumerate(objetivos)]
    src.hide_set(True); src.hide_render = True
    RESUMEN.append(f"{nombre}: tamaño {tuple(round(x, 3) for x in src.dimensions)}, tris {tris(src)} -> {[tris(o) for o in lods]}")
    exportar(lods, salida, anim=False)
    vista(nombre, lods)


def g_carretilla():
    estatico("Carretilla", P("Carretilla", "Carretilla_final.fbx"), P("Carretilla", "Carretilla-color.png"),
             P("Carretilla", "Animados", "Carretilla_LOD.fbx"), [4000, 1600, 600])


def g_nido():
    estatico("Nido", P("Nido", "Nido_final.fbx"), P("Nido", "Nido_color.png"), P("Nido", "Animados", "Nido_LOD.fbx"), [5000, 2000, 800])


for g in GRUPOS:
    globals()["g_" + g]()
print("RESUMEN")
for r in RESUMEN:
    print("  ", r)
print("PROPS_OK")
