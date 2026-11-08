"""Rig + animacion Idle_Viento de los 5 girasoles (Blender CLI).

blender -b -P rig_girasoles.py -- [1 2 3 4 5] [renders=<carpeta>]

Por que asi: cada malla son 34-51 islas sueltas (petalos, hojas, pasto, tallo) que se tocan o se
atraviesan. Para que ninguna pieza se separe, los pesos salen de un campo CONTINUO en el espacio
(dos vertices en el mismo punto reciben el mismo peso aunque sean de islas distintas):
  - tallo: cadena de huesos centrada en el tallo (puntos guia re-centrados sobre la malla);
  - hojas: grupos de vertices lejos del eje del tallo, mezclados con el campo del tallo segun la
    distancia al eje (en la union con el tallo pesan igual que el tallo);
  - cabezas: islas amarillas/cafe (por color de textura) rigidas a su hueso Cabeza;
  - pasto: cada brizna pesa 100% Raiz en su base y se dobla hacia la punta.
La malla no se toca: solo se aplica la rotacion del FBX y se escriben grupos de vertices.
Salida: Animados/Girasol N_anim.blend/.fbx con el clip Idle_Viento (bucle de 120 frames a 30 fps).
"""
import bpy, bmesh, sys, os, math, json
import numpy as np
from mathutils import Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view

BASE = os.path.dirname(os.path.abspath(__file__))
FPS, FR = 30, 120
W = 2 * math.pi / FR

# Puntos guia del tallo (x, y, z) medidos en vistas ortograficas; el script los re-centra sobre la malla.
# Cada cadena termina en el cuello de su cabeza. 'desde' = (cadena, punto) donde nace una rama.
FLORES = {
    1: [dict(n="Tallo", pts=[(-0.03, -0.02, 0.04), (-0.02, -0.02, 0.22), (-0.01, -0.01, 0.42), (0.0, 0.0, 0.64)],
             cabeza=("Cabeza", (0.0, 0.0, 0.82), 0.22))],  # boton: textura poco amarilla, centro/radio fijos
    2: [dict(n="Tallo", pts=[(-0.03, 0.06, 0.04), (-0.02, 0.07, 0.25), (-0.01, 0.07, 0.48), (0.0, 0.06, 0.66), (0.0, 0.04, 0.78)],
             cabeza=("Cabeza", (-0.02, -0.08, 0.85)))],
    3: [dict(n="Tallo", pts=[(-0.01, 0.05, 0.04), (-0.01, 0.06, 0.25), (-0.01, 0.06, 0.48), (-0.01, 0.05, 0.66), (-0.01, 0.03, 0.80)],
             cabeza=("Cabeza", (-0.01, -0.1, 0.86)))],
    4: [dict(n="Tallo", pts=[(0.0, 0.05, 0.04), (0.0, 0.05, 0.2), (0.0, 0.05, 0.42), (0.0, 0.04, 0.62), (0.0, 0.02, 0.76)],
             cabeza=("Cabeza", (0.0, -0.05, 0.85))),
        dict(n="TalloIzq", desde=(0, 1), pts=[(-0.12, 0.04, 0.32), (-0.24, 0.02, 0.43), (-0.32, 0.0, 0.5)],
             cabeza=("CabezaIzq", (-0.38, -0.06, 0.52))),
        dict(n="TalloDer", desde=(0, 1), pts=[(0.12, 0.04, 0.3), (0.22, 0.03, 0.4), (0.29, 0.0, 0.47)],
             cabeza=("CabezaDer", (0.34, -0.06, 0.5)))],
    5: [dict(n="Tallo", pts=[(-0.12, 0.08, 0.04), (-0.13, 0.08, 0.2), (-0.15, 0.08, 0.38), (-0.16, 0.07, 0.52),
                             (-0.12, 0.06, 0.65), (-0.04, 0.05, 0.72)],
             cabeza=("Cabeza", (0.12, -0.06, 0.75)))],
}

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
flores = [int(a) for a in argv if a.isdigit()] or [1, 2, 3, 4, 5]
renders = next((a.split("=", 1)[1] for a in argv if a.startswith("renders=")), None)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def seg_proj(P, h, t):
    d = t - h
    L2 = float(d @ d)
    u = np.clip(((P - h) @ d) / L2, 0.0, 1.0)
    return u, np.linalg.norm(P - (h + u[:, None] * d), axis=1)


def islas(bm):
    isl = np.full(len(bm.verts), -1)
    k = 0
    for v in bm.verts:
        if isl[v.index] >= 0:
            continue
        st = [v]; isl[v.index] = k
        while st:
            a = st.pop()
            for e in a.link_edges:
                b = e.other_vert(a)
                if isl[b.index] < 0:
                    isl[b.index] = k; st.append(b)
        k += 1
    return isl, k


def color_vertices(me, img_path):
    img = bpy.data.images.load(img_path)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(h, w, 4)
    uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    vi = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", vi)
    xs = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int); ys = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
    c = px[ys, xs, :3]
    acc = np.zeros((len(me.vertices), 3)); cnt = np.zeros(len(me.vertices))
    np.add.at(acc, vi, c); np.add.at(cnt, vi, 1)
    return acc / np.maximum(cnt, 1)[:, None], img


def snap(p, t, P, amarillo):
    """Re-centra un punto guia en la seccion del tallo con una transformada de Hough de circulos:
    el anillo del tallo da un radio constante alrededor del centro; las hojas que cruzan la seccion no."""
    d = P - p; a = d @ t
    m = (np.abs(a) < 0.01) & (np.linalg.norm(d - a[:, None] * t, axis=1) < 0.08)
    if m.sum() < 8 or amarillo[m].mean() > 0.05:
        return p.copy(), 0.03
    e1 = np.cross(t, [0, 0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(t, e1)
    Q = np.c_[d[m] @ e1, d[m] @ e2]
    best = (-1, 0, 0, 0.03)
    for cx in np.arange(-0.03, 0.0301, 0.0025):
        for cy in np.arange(-0.03, 0.0301, 0.0025):
            r = np.linalg.norm(Q - [cx, cy], axis=1)
            h, ed = np.histogram(r, bins=np.arange(0.012, 0.073, 0.003))
            k = int(np.argmax(h))
            sc = h[k] - 0.02 * np.hypot(cx, cy) * 1000  # leve preferencia por la guia
            if sc > best[0]:
                best = (sc, cx, cy, (ed[k] + ed[k + 1]) / 2)
    _, cx, cy, r = best
    return p + cx * e1 + cy * e2, float(r)


def construir(num):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS; sc.frame_start = 0; sc.frame_end = FR
    bpy.ops.import_scene.fbx(filepath=f"{BASE}/Finales/Girasol {num}.fbx")
    obj = [o for o in bpy.data.objects if o.type == "MESH"][0]
    for o in list(bpy.data.objects):
        if o != obj:
            bpy.data.objects.remove(o)
    obj.name = obj.data.name = f"Girasol_{num}"
    bpy.context.view_layer.objects.active = obj; obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    me = obj.data
    nv = len(me.vertices)
    P = np.empty(nv * 3); me.vertices.foreach_get("co", P); P = P.reshape(-1, 3)
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    isl, nisl = islas(bm)
    edges = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
    abiertos = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()

    # material limpio con la textura
    col, img = color_vertices(me, f"{BASE}/Textura/Girasol {num}.png")
    mat = bpy.data.materials.new(f"Girasol_{num}"); mat.use_nodes = True
    nt = mat.node_tree; bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
    ti = nt.nodes.new("ShaderNodeTexImage"); ti.image = img
    nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    me.materials.clear(); me.materials.append(mat)

    amar_v = (col[:, 0] > col[:, 1] * 1.08)
    # suaviza el color sobre la topologia: manchas amarillas sueltas en tallo/hojas no cuentan
    af = amar_v.astype(float); deg = np.bincount(edges.ravel(), minlength=nv) + 1.0
    for _ in range(4):
        acc = af.copy(); np.add.at(acc, edges[:, 0], af[edges[:, 1]]); np.add.at(acc, edges[:, 1], af[edges[:, 0]]); af = acc / deg
    amar_v = af > 0.6
    # isla rigida de cabeza solo si es amarilla Y compacta (en G4 una isla une base, tallos y petalos)
    isl_amar = np.array([amar_v[isl == k].mean() > 0.5 and np.ptp(P[isl == k], axis=0).max() < 0.5 for k in range(nisl)])
    amarillo = isl_amar[isl]
    zmin = np.array([P[isl == k, 2].min() for k in range(nisl)]); zmax = np.array([P[isl == k, 2].max() for k in range(nisl)])
    ext = np.array([np.ptp(P[isl == k, :2], axis=0).max() for k in range(nisl)])
    isl_base = (zmax < 0.1) & (ext > 0.3)
    isl_pasto = (zmax < 0.22) & ~isl_base & ~isl_amar
    base_v, pasto_v = isl_base[isl], isl_pasto[isl]
    zb = float(zmax[isl_base].max()) if isl_base.any() else 0.08   # la base de G4 esta fundida con el tallo
    nrm_v = np.empty(nv * 3); me.vertices.foreach_get("normal", nrm_v); nrm_v = nrm_v.reshape(-1, 3)
    from mathutils.kdtree import KDTree
    kd = KDTree(nv)
    for i, p in enumerate(P):
        kd.insert(Vector(p), i)
    kd.balance()
    pares = set()
    for i, p in enumerate(P):
        for (_, j, _) in kd.find_range(Vector(p), 0.003):
            if j > i and isl[j] != isl[i]:
                pares.add((i, j))
    pares = np.array(sorted(pares)) if pares else np.zeros((0, 2), int)
    verde = ~amarillo & ~base_v & ~pasto_v

    # ------------------------------------------------ cadenas de tallo + cabezas
    cfg = FLORES[num]
    huesos = [dict(n="Raiz", h=np.array([0, 0, 0.0]), t=np.array([0, 0, 0.05]), p=None, k="raiz")]
    juntas = []          # por cadena: lista de nombres de hueso por segmento
    cabezas = []
    cen_cfg = np.array([c["cabeza"][1] for c in cfg])
    # asigna islas amarillas a la cabeza mas cercana
    cab_de_isla = {}
    for k in np.where(isl_amar)[0]:
        c = P[isl == k].mean(axis=0)
        cab_de_isla[k] = int(np.argmin(np.linalg.norm(cen_cfg - c, axis=1)))
    for ci, c in enumerate(cfg):
        pts = [np.array(p, float) for p in c["pts"]]
        if "desde" in c:
            pc, pj = c["desde"]
            pts = [huesos_pos[pc][pj]] + pts
        Pv = P[verde]; Am = amarillo[verde]
        snapped, radios = [], []
        for i, p in enumerate(pts):
            if "desde" in c and i == 0:
                snapped.append(p); radios.append(radios_cad[c["desde"][0]][c["desde"][1]]); continue
            a = pts[max(i - 1, 0)]; b = pts[min(i + 1, len(pts) - 1)]
            t = (b - a) / np.linalg.norm(b - a)
            q, r = snap(p, t, Pv, Am)
            if i == 0:
                q[2] = zb
            snapped.append(q); radios.append(r)
        # cuello: primer punto del tallo que toca la cabeza. Ahi pivota la cabeza; el tallo que sigue por
        # detras de los petalos (hasta 'tope') queda dentro del hueso Cabeza y se mueve con ella.
        sel = np.array([cab_de_isla.get(k, -1) == ci for k in isl])
        dcab = np.linalg.norm(P[:, None, :] - cen_cfg[None], axis=2)
        selv = amar_v & ~base_v & ~pasto_v & (np.argmin(dcab, axis=1) == ci) & (dcab[:, ci] < 0.45)
        Yp = P[selv]; tope = snapped[-1]; cuello = None
        for i in range(len(snapped) - 1):
            for s in np.linspace(0, 1, 21):
                q = snapped[i] + (snapped[i + 1] - snapped[i]) * s
                r = radios[i] + (radios[i + 1] - radios[i]) * s
                if np.linalg.norm(q - snapped[0]) > 0.1 and np.linalg.norm(Yp - q, axis=1).min() < min(r, 0.04) + 0.015:
                    cuello = (i, q, r); break
            if cuello:
                break
        if cuello:
            i, q, r = cuello
            snapped, radios = snapped[:i + 1], radios[:i + 1]
            if np.linalg.norm(q - snapped[-1]) > 0.03 or len(snapped) < 2:
                snapped.append(q); radios.append(r)
        if ci == 0:
            huesos_pos, radios_cad = [], []
        huesos_pos.append(snapped); radios_cad.append(radios)
        nombres = []
        for i in range(len(snapped) - 1):
            if i == 0 and "desde" in c:
                par = juntas[c["desde"][0]][c["desde"][1] - 1]
            else:
                par = nombres[-1] if nombres else "Raiz"
            nm = f"{c['n']}_{i + 1}"
            huesos.append(dict(n=nm, h=snapped[i], t=snapped[i + 1], p=par, k="tallo", r=max(radios[i], radios[i + 1]),
                               cad=ci, s=(i + 1) / (len(snapped) - 1)))
            nombres.append(nm)
        juntas.append(nombres)
        cen = P[selv].mean(axis=0)
        R = float(np.percentile(np.linalg.norm(P[selv] - cen, axis=1), 90))
        if len(c["cabeza"]) > 2:
            cen, R = np.array(c["cabeza"][1], float), float(c["cabeza"][2])
        cabezas.append(dict(n=c["cabeza"][0], cen=cen, R=R, sel=sel))
        cu = snapped[-1]
        if np.linalg.norm(tope - cu) > 0.03:
            tc = tope
        else:
            dcu = cen - cu; tc = cu + dcu / max(np.linalg.norm(dcu), 1e-6) * 0.1
        huesos.append(dict(n=c["cabeza"][0], h=cu, t=tc, p=nombres[-1], k="cabeza", cad=ci, r=radios[-1]))
    H = {b["n"]: b for b in huesos}
    tallos = [b for b in huesos if b["k"] in ("tallo", "cabeza")]
    hijos = {b["n"]: [c["n"] for c in tallos if c["p"] == b["n"] and c["k"] == "tallo"] for b in tallos if b["k"] == "tallo"}
    for b in [x for x in tallos if x["k"] == "tallo"]:
        if not hijos[b["n"]]:
            hijos[b["n"]] = [cfg[b["cad"]]["cabeza"][0]]

    # ------------------------------------------------ campo del tallo
    nb_t = len(tallos)
    U = np.zeros((nv, nb_t)); D = np.zeros((nv, nb_t))
    for j, b in enumerate(tallos):
        U[:, j], D[:, j] = seg_proj(P, b["h"], b["t"])
    jn = np.argmin(D, axis=1)
    dax = D[np.arange(nv), jn]
    rax = np.array([tallos[j]["r"] for j in jn])
    names = [b["n"] for b in huesos]
    idx = {n: i for i, n in enumerate(names)}
    S = np.zeros((nv, len(huesos)))
    for v in range(nv):
        b = tallos[jn[v]]; u = U[v, jn[v]]
        if b["k"] == "cabeza":
            if u < 0.5:
                S[v, idx[b["n"]]] += 0.5 + u; S[v, idx[b["p"]]] += 0.5 - u
            else:
                S[v, idx[b["n"]]] += 1.0
        elif u < 0.5:
            S[v, idx[b["n"]]] += 0.5 + u; S[v, idx[b["p"]]] += 0.5 - u
        else:
            hs = hijos[b["n"]]
            if len(hs) > 1:
                dd = [np.linalg.norm(P[v] - H[h]["t"]) for h in hs]
                hc = hs[int(np.argmin(dd))]
            else:
                hc = hs[0]
            S[v, idx[b["n"]]] += 1.5 - u; S[v, idx[hc]] += u - 0.5

    # influencia de cabeza: islas amarillas rigidas, verde cercano por distancia
    hcab = np.zeros((nv, len(cabezas)))
    for ci, cb in enumerate(cabezas):
        dist = np.linalg.norm(P - cb["cen"], axis=1)
        # islas verdes pequenas sueltas dentro de la cabeza (sepalos) tambien rigidas
        for k in range(nisl):
            m = isl == k
            if not isl_amar[k] and not isl_base[k] and m.sum() < 500 and (dist[m] < 0.8 * cb["R"]).all():
                hcab[m, ci] = 1.0
        hcab[cb["sel"], :] = 0.0
        hcab[cb["sel"], ci] = 1.0
        # petalos dentro de islas fundidas: rigidos por vertice
        dcab = np.linalg.norm(P[:, None, :] - np.array([c["cen"] for c in cabezas])[None], axis=2)
        fus = amar_v & ~amarillo & ~base_v & ~pasto_v & (np.argmin(dcab, axis=1) == ci) & (dist < 1.05 * cb["R"])
        hcab[fus, :] = 0.0; hcab[fus, ci] = 1.0
        # parte verde de la cabeza (dorso, sepalos, boton de G1, tallo detras de los petalos)
        esf = ~amar_v & ~base_v & ~pasto_v & (np.argmin(dcab, axis=1) == ci) & (dist < 0.7 * cb["R"])
        hcab[esf, :] = 0.0; hcab[esf, ci] = 1.0
    hsum = hcab.sum(axis=1)
    rigida_isla = np.zeros(nv, bool)
    for cb in cabezas:
        rigida_isla |= cb["sel"]

    # ------------------------------------------------ hojas
    lejos_cab = np.min(np.stack([np.linalg.norm(P - c["cen"], axis=1) / c["R"] for c in cabezas]), axis=0) > 0.9
    cand = verde & ~amar_v & (hsum == 0) & lejos_cab & (P[:, 2] > zb + 0.03) & (dax > rax * 1.3 + 0.012)
    adj = [[] for _ in range(nv)]
    for a, b in np.vstack([edges, pares]):
        if cand[a] and cand[b]:
            adj[a].append(b); adj[b].append(a)
    lab = np.full(nv, -1); hojas = []
    for v in np.where(cand)[0]:
        if lab[v] >= 0:
            continue
        st = [v]; lab[v] = len(hojas); comp = []
        while st:
            a = st.pop(); comp.append(a)
            for b in adj[a]:
                if lab[b] < 0:
                    lab[b] = len(hojas); st.append(b)
        comp = np.array(comp)
        if len(comp) < 80 or np.ptp(P[comp], axis=0).max() < 0.1:
            lab[comp] = -2
            continue
        hojas.append(comp)
    L = np.zeros((nv, len(huesos) + 2 * len(hojas)))
    L[:, :len(huesos)] = S
    info_hojas = []
    for hk, comp in enumerate(hojas):
        a = P[comp[np.argmin(dax[comp])]]
        tip = P[comp[np.argmax(np.linalg.norm(P[comp] - a, axis=1))]]
        d = tip - a; ln = np.linalg.norm(d); d /= ln
        X = P[comp] - P[comp].mean(axis=0)
        nrm = np.linalg.svd(X, full_matrices=False)[2][2]
        par_j = int(np.argmin([seg_proj(a[None], b["h"], b["t"])[1][0] for b in tallos]))
        n1, n2 = f"Hoja_{hk + 1}_A", f"Hoja_{hk + 1}_B"
        mid = a + d * ln * 0.5
        huesos.append(dict(n=n1, h=a, t=mid, p=tallos[par_j]["n"], k="hoja", eje=np.cross(d, nrm), dir=d, fase=hk * 2.39996))
        huesos.append(dict(n=n2, h=mid, t=tip, p=n1, k="hoja", eje=np.cross(d, nrm), dir=d, fase=hk * 2.39996 - 0.9))
        info_hojas.append((n1, len(comp), round(float(ln), 3)))
        m = lab == hk
        # todos los vertices de la hoja (incluida la parte junto al tallo, fuera de 'cand') via vecindad espacial
        u = np.clip(((P[m] - a) @ d) / ln, 0, 1)
        umbral = rax[m] * 1.3 + 0.012  # mismo umbral que cand: en el borde del grupo f=0, continuo con el tallo
        f = smooth(umbral, umbral + 0.10, dax[m])
        w2 = smooth(0.3, 0.8, u)
        ib1, ib2 = len(huesos) - 2, len(huesos) - 1
        L[m, :] *= (1 - f)[:, None]
        L[m, ib1] += f * (1 - w2); L[m, ib2] += f * w2
    names = [b["n"] for b in huesos]
    idx = {n: i for i, n in enumerate(names)}

    # cabezas encima
    Wt = L * (1 - np.clip(hsum, 0, 1))[:, None]
    for ci, cb in enumerate(cabezas):
        Wt[:, idx[cb["n"]]] += hcab[:, ci]

    # pasto y base
    huesos.append(dict(n="Pasto_Izq", h=np.array([-0.15, 0, 0.02]), t=np.array([-0.15, 0, 0.1]), p="Raiz", k="pasto", fase=0.0))
    huesos.append(dict(n="Pasto_Der", h=np.array([0.15, 0, 0.02]), t=np.array([0.15, 0, 0.1]), p="Raiz", k="pasto", fase=2.1))
    names = [b["n"] for b in huesos]; idx = {n: i for i, n in enumerate(names)}
    Wf = np.zeros((nv, len(huesos))); Wf[:, :Wt.shape[1]] = Wt
    wR = smooth(zb + 0.035, zb + 0.005, P[:, 2])
    Wf = Wf * (1 - wR)[:, None]; Wf[:, idx["Raiz"]] += wR
    Wf[base_v] = 0; Wf[base_v, idx["Raiz"]] = 1
    # pasto: campo continuo en el espacio (briznas que se cruzan pesan igual en el cruce).
    # Altura de la superficie de la base bajo cada punto -> por debajo 100% Raiz; se dobla hacia arriba.
    tapa = base_v & (nrm_v[:, 2] > 0.5)
    Pb = P[tapa]
    def z_base(Q):
        if not len(Pb):
            return np.full(len(Q), zb)
        out = np.empty(len(Q))
        for a in range(0, len(Q), 1000):
            d = np.linalg.norm(Q[a:a + 1000, None, :2] - Pb[None, :, :2], axis=2)
            w = np.exp(-(d / 0.02) ** 2) + 1e-12
            out[a:a + 1000] = (w * Pb[None, :, 2]).sum(axis=1) / w.sum(axis=1)
        return out
    pm = pasto_v
    z0 = z_base(P[pm]) + 0.004
    g = smooth(z0, z0 + 0.09, P[pm, 2])
    lado = smooth(-0.12, 0.12, P[pm, 0])
    Wf[pm] = 0
    Wf[pm, idx["Raiz"]] = 1 - g
    Wf[pm, idx["Pasto_Izq"]] = g * (1 - lado); Wf[pm, idx["Pasto_Der"]] = g * lado

    # reparacion: donde quedan saltos de peso (entre aristas o entre islas que se tocan) se difunden los
    # pesos sobre la topologia (aristas + contactos) en un radio de 4 cm, con cabezas rigidas y base fijas.
    Gr = np.vstack([edges] + [pares] * 8) if len(pares) else edges  # contactos con mas peso: se mueven juntos
    Lg = np.linalg.norm(P[Gr[:, 0]] - P[Gr[:, 1]], axis=1)
    # fijos: base y cabezas de islas compactas. Los petalos fundidos (G4) pueden difundirse en la union
    # entre dos cabezas. Lo que toca una pieza fija copia su peso y queda fijo (sin grieta en el contacto).
    fijo = base_v | rigida_isla
    if len(pares):
        for a, b in pares:
            if fijo[a] and not fijo[b]:
                Wf[b] = Wf[a]
            elif fijo[b] and not fijo[a]:
                Wf[a] = Wf[b]
        pf = fijo[pares[:, 0]] | fijo[pares[:, 1]]
        fijo[pares[pf].ravel()] = True
    for ronda in range(2):
        dif = np.abs(Wf[Gr[:, 0]] - Wf[Gr[:, 1]]).sum(axis=1)
        malas = Gr[(dif > 0.05) & (dif / np.maximum(Lg, 1e-4) > 25)]
        if not len(malas):
            break
        libre = np.zeros(nv, bool)
        for v in np.unique(malas):
            for (_, j, _) in kd.find_range(Vector(P[v]), 0.04):
                libre[j] = True
        libre &= ~fijo
        dg_ = np.bincount(Gr.ravel(), minlength=nv).astype(float)
        for _ in range(400):
            acc = np.zeros_like(Wf)
            np.add.at(acc, Gr[:, 0], Wf[Gr[:, 1]]); np.add.at(acc, Gr[:, 1], Wf[Gr[:, 0]])
            Wf[libre] = acc[libre] / np.maximum(dg_[libre], 1)[:, None]
        print(f"   reparacion ronda {ronda}: saltos={len(malas)} vertices_difundidos={int(libre.sum())}")

    # limpieza: max 4 influencias, normalizar
    orden = np.argsort(Wf, axis=1)
    np.put_along_axis(Wf, orden[:, :-4], 0.0, axis=1)
    Wf[Wf < 0.01] = 0.0
    assert (Wf.sum(axis=1) > 0).all(), "vertices sin peso"
    Wf /= Wf.sum(axis=1, keepdims=True)

    # ------------------------------------------------ armature
    arm_d = bpy.data.armatures.new("Girasol_Rig"); arm = bpy.data.objects.new("Girasol_Rig", arm_d)
    sc.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for b in huesos:
        eb = arm_d.edit_bones.new(b["n"]); eb.head = Vector(b["h"]); eb.tail = Vector(b["t"])
        if b["p"]:
            eb.parent = arm_d.edit_bones[b["p"]]
            eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    arm_d.display_type = "STICK"
    for j, n in enumerate(names):
        ids = np.where(Wf[:, j] > 0)[0]
        if len(ids):
            g = obj.vertex_groups.new(name=n)
            for i in ids:
                g.add([int(i)], float(Wf[i, j]), "REPLACE")
    obj.parent = arm
    mod = obj.modifiers.new("Armature", "ARMATURE"); mod.object = arm

    # ------------------------------------------------ animacion Idle_Viento
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    X, Y = Vector((1, 0, 0)), Vector((0, 0, 1)).cross(Vector((1, 0, 0)))  # Y = (0,1,0)
    def local(pb, w):
        return (pb.bone.matrix_local.to_3x3().inverted() @ Vector(w)).normalized()
    deg = math.radians
    for f in range(FR + 1):
        for pb in arm.pose.bones:
            b = H.get(pb.name) or next(x for x in huesos if x["n"] == pb.name)
            q = Quaternion()
            if b["k"] == "tallo":
                ph = b["cad"] * 1.3; s = b["s"]
                ay = 1.9 * (0.3 + 0.75 * math.sin(W * f - 0.9 * s + ph) + 0.25 * math.sin(2 * W * f - 1.6 * s + 0.7 + ph))
                ax = 1.1 * (0.65 * math.sin(W * f + 1.9 - 0.9 * s + ph) + 0.35 * math.sin(3 * W * f + 0.4 + ph))
                q = Quaternion(local(pb, Y), deg(ay)) @ Quaternion(local(pb, X), deg(ax))
            elif b["k"] == "cabeza":
                ph = b["cad"] * 1.3
                ay = 2.6 * math.sin(W * f - 1.5 + ph)
                ax = 2.2 * math.sin(2 * W * f - 0.9 + ph)
                tw = 2.0 * math.sin(W * f + 2.2 + ph)
                q = Quaternion(local(pb, Y), deg(ay)) @ Quaternion(local(pb, X), deg(ax)) @ Quaternion((0, 1, 0), deg(tw))
            elif b["k"] == "hoja":
                fa = b["fase"]; amp = 1.0 if pb.name.endswith("_A") else 1.3
                be = amp * (4 * math.sin(3 * W * f + fa) + 1.5 * math.sin(5 * W * f + 1.3 * fa))
                tw = amp * 2.5 * math.sin(2 * W * f + fa + 1.0)
                q = Quaternion(local(pb, b["eje"]), deg(be)) @ Quaternion(local(pb, b["dir"]), deg(tw))
            elif b["k"] == "pasto":
                # traslacion (no rotacion): cada brizna se inclina desde su propia base, sin pivote lejano
                fa = b["fase"]
                off = Vector((0.0045 * math.sin(4 * W * f + fa) + 0.0015,
                              0.003 * math.sin(3 * W * f + fa + 1) + 0.001 * math.sin(6 * W * f), 0))
                pb.location = pb.bone.matrix_local.to_3x3().inverted() @ off
                pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=f)
    act = arm.animation_data.action; act.name = "Idle_Viento"
    for fc in (act.fcurves if hasattr(act, "fcurves") else [c for l in act.layers for s in l.strips for cb in s.channelbags for c in cb.fcurves]):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

    # ------------------------------------------------ verificacion: grietas entre islas y estiramiento
    d0 = np.linalg.norm(P[pares[:, 0]] - P[pares[:, 1]], axis=1) if len(pares) else np.zeros(0)
    e0 = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
    grieta, estira, peor = 0.0, 0.0, None
    largas = e0 > 0.004; peor_e = None; p99 = 0.0
    dg = bpy.context.evaluated_depsgraph_get()
    for f in range(0, FR + 1, 5):
        sc.frame_set(f)
        eo = obj.evaluated_get(dg); m2 = eo.to_mesh()
        Q = np.empty(nv * 3); m2.vertices.foreach_get("co", Q); Q = Q.reshape(-1, 3)
        eo.to_mesh_clear()
        Qw = Q
        if len(pares):
            dd = np.abs(np.linalg.norm(Qw[pares[:, 0]] - Qw[pares[:, 1]], axis=1) - d0)
            if dd.max() > grieta:
                grieta = float(dd.max()); a, b = pares[np.argmax(dd)]
                peor = (f, int(isl[a]), int(isl[b]), np.round(P[a], 3).tolist(),
                        {names[j]: round(float(Wf[a, j]), 2) for j in np.nonzero(Wf[a])[0]},
                        {names[j]: round(float(Wf[b, j]), 2) for j in np.nonzero(Wf[b])[0]})
        e1 = np.linalg.norm(Qw[edges[:, 0]] - Qw[edges[:, 1]], axis=1)
        rr = np.abs(e1 / np.maximum(e0, 1e-9) - 1); rr[~largas] = 0
        p99 = max(p99, float(np.percentile(rr[largas], 99.9)))
        if rr.max() > estira:
            estira = float(rr.max()); a, b = edges[np.argmax(rr)]
            peor_e = (f, int(isl[a]), np.round(P[a], 3).tolist(), round(float(e0[np.argmax(rr)]) * 1000, 2),
                      {names[j]: round(float(Wf[a, j]), 2) for j in np.nonzero(Wf[a])[0]},
                      {names[j]: round(float(Wf[b, j]), 2) for j in np.nonzero(Wf[b])[0]})
    print(f"GIRASOL {num}: islas={nisl} amarillas={int(isl_amar.sum())} pasto={int(isl_pasto.sum())} base={int(isl_base.sum())} "
          f"huesos={len(huesos)} hojas={info_hojas} bordes_abiertos={abiertos} (sin cambio, malla intacta)")
    print(f"   cabezas={[(c['n'], np.round(c['cen'], 3).tolist(), round(c['R'], 3)) for c in cabezas]}")
    print(f"   tallo={[(b['n'], np.round(b['h'], 3).tolist(), round(b['r'], 3)) for b in tallos]}")
    print(f"   VERIFICACION: pares de contacto entre islas={len(pares)} separacion_max={grieta * 1000:.2f} mm  "
          f"estiramiento aristas>4mm: max={estira * 100:.2f}% p99.9={p99 * 100:.2f}%")
    print(f"   peor contacto: {peor}")
    print(f"   peor arista: {peor_e}")

    if renders:
        render_hojas(num, obj, arm, huesos, Wf, names, renders)

    sc.frame_set(0)
    out = f"{BASE}/Animados"; os.makedirs(out, exist_ok=True)
    blend, fbx = f"{out}/Girasol {num}_anim.blend", f"{out}/Girasol {num}_anim.fbx"
    bpy.ops.export_scene.fbx(
        filepath=fbx, object_types={"ARMATURE", "MESH"}, use_selection=False,
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()
    if os.path.exists(blend + "1"):
        os.remove(blend + "1")


def render_hojas(num, obj, arm, huesos, Wf, names, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.render.resolution_x = sc.render.resolution_y = 400
    cam = bpy.data.objects.new("cam_tmp", bpy.data.cameras.new("cam_tmp")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"; cam.data.ortho_scale = 1.2
    vistas = {"F": ((0, -3, 0.55), (math.pi / 2, 0, 0)), "S": ((3, 0, 0.55), (math.pi / 2, 0, math.pi / 2))}
    # colores de pesos: mezcla de colores por hueso
    rng = np.random.default_rng(num)
    pal = rng.uniform(0.15, 1.0, (len(names), 3)); pal[names.index("Raiz")] = (0.35, 0.35, 0.35)
    cv = np.clip(Wf @ pal, 0, 1)
    ca = obj.data.color_attributes.new("Pesos_tmp", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.c_[cv, np.ones(len(cv))].ravel().astype(np.float32))
    obj.data.color_attributes.active_color = ca
    lineas = {}
    sc.frame_set(0)
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
    ad = arm.animation_data; act = ad.action; ad.action = None
    bpy.context.view_layer.update()
    for v, (loc, rot) in vistas.items():
        cam.location = loc; cam.rotation_euler = rot
        bpy.context.view_layer.update()
        sc.display.shading.color_type = "VERTEX"
        sc.render.filepath = f"{carpeta}/g{num}_pesos_{v}.png"; bpy.ops.render.render(write_still=True)
        seg = []
        for b in huesos:
            a = world_to_camera_view(sc, cam, arm.matrix_world @ Vector(b["h"])); c = world_to_camera_view(sc, cam, arm.matrix_world @ Vector(b["t"]))
            seg.append([a.x * 400, (1 - a.y) * 400, c.x * 400, (1 - c.y) * 400, b["k"]])
        lineas[f"g{num}_pesos_{v}"] = seg
    obj.data.color_attributes.remove(obj.data.color_attributes["Pesos_tmp"])
    ad.action = act
    if hasattr(act, "slots") and act.slots:
        ad.action_slot = act.slots[0]
    sc.display.shading.color_type = "TEXTURE"
    cam.location, cam.rotation_euler = vistas["F"]
    for f in (0, 30, 60, 90):
        sc.frame_set(f)
        sc.render.filepath = f"{carpeta}/g{num}_f{f:03d}.png"; bpy.ops.render.render(write_still=True)
    cam.location, cam.rotation_euler = vistas["S"]
    for f in (15, 75):
        sc.frame_set(f)
        sc.render.filepath = f"{carpeta}/g{num}_S{f:03d}.png"; bpy.ops.render.render(write_still=True)
    with open(f"{carpeta}/g{num}_huesos.json", "w") as fh:
        json.dump(lineas, fh)
    bpy.data.objects.remove(cam)


for n in flores:
    construir(n)
