"""Medidas del brazo a partir de la malla en pose T (se usa desde rig_granjero_v2.py y para revisar).
Cortes transversales exactos (intersección de las aristas con planos x = cte), siguiendo el tubo del brazo desde la punta
de la mano hacia dentro. Devuelve axila, hombro, codo, muñeca, punta, pulgar y el radio del tubo a lo largo del brazo."""
import numpy as np
DEBUG = False


def cortes(V, E, xs_list, sg):
    """Puntos (y, z) de la sección en cada x (lado sg)."""
    a, b = V[E[:, 0]], V[E[:, 1]]
    xa, xb = a[:, 0] * sg, b[:, 0] * sg
    out = []
    for xs in xs_list:
        m = (np.minimum(xa, xb) <= xs) & (np.maximum(xa, xb) > xs)
        if not m.any():
            out.append(np.zeros((0, 2))); continue
        t = ((xs - xa[m]) / (xb[m] - xa[m]))[:, None]
        p = a[m] + (b[m] - a[m]) * t
        out.append(p[:, 1:3])
    return out


# Revisión a mano, personaje por personaje (vistas de frente y desde arriba en pose T): x de la muñeca, justo donde
# acaba la manga o el puño y empieza la mano. Lo demás (axila, hombro, codo, pulgar) sale de la malla; el codo (x) también
# se puede fijar (por defecto, a media distancia entre hombro y muñeca).
AJUSTES = {
    'Paco': dict(wrist=0.300),          # manga remangada hasta 0.29, manopla grande
    # Lola (malla nueva 0.9.3): manga remangada 0.19-0.234; con el codo a media distancia (0.197) la vuelta de la manga
    # caía justo en la bisagra y se aplastaba al doblar (Dance, Taunt). Codo al final de la vuelta: dobla la piel.
    # Arreglos 0.9.3 de rig_granjero_v2.py (bajo el brazo, sisa, capas de ropa, manos fuera del cuerpo): sólo Lola por ahora.
    'Lola': dict(wrist=0.282, elbow=0.222, bajo_brazo=True, sisa=True, capas=True, sin_choques=True),
    'AbueloTomas': dict(wrist=0.298),   # puño hasta 0.285
    'Rosa': dict(wrist=0.277),          # puño hasta 0.265
    'Chema': dict(wrist=0.282),         # puño hasta 0.265
}


def medir_brazo(V, E, sg, armZ, H, log=print, name=None):
    x = V[:, 0] * sg
    tip = float(x.max())
    step = 0.003
    xs_list = np.arange(tip - 0.006, 0.0, -step)
    S = cortes(V, E, xs_list, sg)
    # Seguimiento del tubo con una caja (y, z) que puede crecer poco por corte: el pelo o el cuerpo que aparecen de golpe
    # alrededor quedan fuera. Axila: aparece cuerpo justo debajo del brazo (franja estrecha bajo su borde inferior).
    box = None          # [ylo, yhi, zlo, zhi]
    prof = []           # (x, cy, cz, ry, rz)
    armpit = None
    G = 0.012
    for xs, P in zip(xs_list, S):
        if box is None:
            sel = P[np.abs(P[:, 1] - armZ) < 0.12 * H] if len(P) else P
            if len(sel) < 3: continue
            box = [sel[:, 0].min(), sel[:, 0].max(), sel[:, 1].min(), sel[:, 1].max()]
        else:
            if len(P):
                below = P[(P[:, 1] < box[2] - 0.03) & (P[:, 1] > box[2] - 0.07 * H)
                          & (P[:, 0] > box[0] - 0.02) & (P[:, 0] < box[1] + 0.02)]
                if len(below) >= 2 and xs < tip - 0.12:
                    armpit = float(xs)
                    if DEBUG: log(f'  AXILA x={xs:.3f} ({len(below)} puntos bajo z={box[2]:.3f})')
                    break
            sel = P[(P[:, 0] > box[0] - G) & (P[:, 0] < box[1] + G) & (P[:, 1] > box[2] - 0.03) & (P[:, 1] < box[3] + G)] if len(P) else P
            if len(sel) >= 6:
                nb = [sel[:, 0].min(), sel[:, 0].max(), sel[:, 1].min(), sel[:, 1].max()]
                # Encoger sólo poco a poco (un corte que pilla un hueco entre piezas no estrecha el tubo de golpe).
                box = [max(nb[0], box[0] - G) if nb[0] < box[0] else min(nb[0], box[0] + 0.004),
                       min(nb[1], box[1] + G) if nb[1] > box[1] else max(nb[1], box[1] - 0.004),
                       nb[2] if nb[2] < box[2] else min(nb[2], box[2] + 0.004),
                       min(nb[3], box[3] + 0.005) if nb[3] > box[3] else max(nb[3], box[3] - 0.004)]
        cy, cz = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
        ry, rz = max(0.012, (box[1] - box[0]) / 2), max(0.012, (box[3] - box[2]) / 2)
        prof.append((xs, cy, cz, ry, rz))
        if DEBUG: log(f'  x={xs:.3f} c=({cy:.3f},{cz:.3f}) r=({ry:.3f},{rz:.3f}) n={len(P)} sel={len(sel)}')
    prof = np.array(prof[::-1])          # de dentro a fuera
    if armpit is None: armpit = float(prof[0, 0])
    # Radio del tubo (media de los semiejes) a lo largo del brazo.
    rad = (prof[:, 3] + prof[:, 4]) / 2

    def at(xq):
        i = int(np.argmin(np.abs(prof[:, 0] - xq)))
        return prof[i]

    near = prof[prof[:, 0] < armpit + 0.05]
    r_sh = float(np.median((near[:, 3] + near[:, 4]) / 2)) if len(near) else float(rad[0])
    sh_x = armpit - 0.55 * r_sh
    pa = at(armpit + 0.02)
    # z del hombro: borde inferior del tubo (fiable) + radio (el borde superior lo puede hinchar el pelo).
    up = prof[(prof[:, 0] > armpit) & (prof[:, 0] < armpit + 0.06)]
    r_up = float(np.median(np.minimum(up[:, 3], up[:, 4]))) if len(up) else r_sh
    shoulder = np.array([sh_x, pa[1], (pa[2] - pa[4]) + r_up])
    L = tip - sh_x
    # Fin de la manga/puño: la última caída brusca del radio yendo hacia la mano.
    rr = (prof[:, 3] + prof[:, 4]) / 2
    cuff = None
    for i in range(len(prof)):
        xi = prof[i, 0]
        if xi < sh_x + 0.35 * L or xi > tip - 0.18 * L: continue
        j = np.where((prof[:, 0] > xi) & (prof[:, 0] < xi + 0.013))[0]
        if len(j) and rr[j].min() < 0.8 * rr[i]: cuff = xi
    # Pulgar: punto más alejado del eje en la mano (desde el puño, sin la punta de los dedos).
    hz0 = (cuff + 0.01) if cuff is not None else sh_x + 0.5 * L
    hz1 = tip - 0.12 * L
    hv = V[(x > hz0) & (x < hz1)]
    best, thumb_tip = -1, None
    for v in hv:
        pr = at(v[0] * sg)
        dd = np.hypot(v[1] - pr[1], v[2] - pr[2])
        if dd > 2.4 * max(pr[3], pr[4]) or abs(v[2] - pr[2]) > 2.0 * pr[4] + 0.02: continue   # ala del sombrero, pelo...
        if dd > best: best, thumb_tip = dd, v.copy()
    # Muñeca: la sección estrecha más lejana entre el puño (o el codo provisional) y el pulgar.
    ex0 = cuff if cuff is not None else sh_x + 0.42 * L
    tx = thumb_tip[0] * sg if thumb_tip is not None else tip - 0.3 * L
    seg = prof[(prof[:, 0] > ex0) & (prof[:, 0] < tx - 0.01)]
    if len(seg) >= 2:
        area = seg[:, 3] * seg[:, 4]
        ok = np.where(area <= area.min() * 1.18)[0]
        wrist_x = float(seg[ok, 0].max())          # el punto estrecho más lejano (el antebrazo es un tubo casi recto)
        wrist_x = min(wrist_x, tx - 0.15 * (tip - tx) - 0.01)
    else:
        wrist_x = (cuff + 0.01) if cuff is not None else tip - 0.38 * L
    wrist_x = float(np.clip(wrist_x, sh_x + 0.5 * L, tip - 0.25 * L))
    if name in AJUSTES and 'wrist' in AJUSTES[name]:
        wrist_x = AJUSTES[name]['wrist']
        # Pulgar otra vez, ya con la muñeca buena: sólo en la mano.
        hv = V[(x > wrist_x) & (x < tip - 0.12 * L)]
        best, thumb_tip = -1, None
        for v in hv:
            pr = at(v[0] * sg)
            dd = np.hypot(v[1] - pr[1], v[2] - pr[2])
            if dd > 2.4 * max(pr[3], pr[4]) or abs(v[2] - pr[2]) > 2.0 * pr[4] + 0.02: continue
            if dd > best: best, thumb_tip = dd, v.copy()
    pw = at(wrist_x)
    wrist = np.array([wrist_x, pw[1], pw[2]])
    elbow_x = sh_x + 0.5 * (wrist_x - sh_x)
    if name in AJUSTES and 'elbow' in AJUSTES[name]:
        elbow_x = AJUSTES[name]['elbow']
    pe = at(elbow_x)
    elbow = np.array([elbow_x, pe[1], pe[2]])
    r_el = float((pe[3] + pe[4]) / 2)
    pt = at(tip - 0.01)
    res = dict(cuff=cuff, armpit=armpit, shoulder=shoulder, elbow=elbow, wrist=wrist, tip=tip, tip_c=np.array([tip, pt[1], pt[2]]),
               thumb_tip=thumb_tip, r_sh=r_sh, r_el=r_el, r_wr=float((pw[3] + pw[4]) / 2), prof=prof)
    log(f'BRAZO {"L" if sg > 0 else "R"}: axila x={armpit:.3f} puño {cuff} hombro {np.round(shoulder, 3)} codo {np.round(elbow, 3)} '
        f'muñeca {np.round(wrist, 3)} punta {tip:.3f} pulgar {None if thumb_tip is None else np.round(thumb_tip, 3)} '
        f'radios hombro {r_sh:.3f} codo {r_el:.3f} muñeca {res["r_wr"]:.3f}')
    return res
