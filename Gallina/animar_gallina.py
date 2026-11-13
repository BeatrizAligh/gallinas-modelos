"""Crea las animaciones de la gallina sobre el rig, por linea de comandos.

    blender --background --disable-autoexec "gallina prueba modelo_rig.blend" ^
            --python animar_gallina.py -- "<salida sin extension>"

Genera <salida>.blend (una Action por clip) y <salida>.fbx (malla + esqueleto + todos los clips).
Todas las animaciones van en el sitio: el desplazamiento real lo pone ChickenMotor.

Convencion de angulos (grados, ejes del rig en reposo; la gallina mira a -Y, su izquierda es +X):
    rx + : lo de arriba cae hacia delante (lo que apunta atras sube)
    ry + : lo de arriba se inclina a su izquierda
    rz + : gira hacia su izquierda
"""
import bpy, sys, os, math
import numpy as np
from mathutils import Euler, Quaternion, Vector
from bpy_extras import anim_utils

out_base = sys.argv[sys.argv.index("--") + 1]
FPS = 30
TAU = math.tau
sin, cos = math.sin, math.cos

CLIPS = {}


def clip(segundos, loop=False):
    def deco(fn):
        CLIPS[fn.__name__] = (fn, int(round(segundos * FPS)), loop)
        return fn
    return deco


# ---------------------------------------------------------------- utilidades de curvas
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def back(x, s=2.6):
    """Llega pasandose y vuelve: el rebote que da vida a los golpes."""
    x = min(1.0, max(0.0, x)) - 1
    return x * x * ((s + 1) * x + s) + 1


def K(t, *pts):
    """Curva por claves (tiempo, valor[, 'o' = llega con rebote])."""
    if t <= pts[0][0]:
        return pts[0][1]
    for a, b in zip(pts, pts[1:]):
        if t <= b[0]:
            x = (t - a[0]) / (b[0] - a[0])
            f = back(x) if len(b) > 2 else ease(x)
            return a[1] + (b[1] - a[1]) * f
    return pts[-1][1]


def pulso(t, t0, dur):
    return sin(math.pi * min(1.0, max(0.0, (t - t0) / dur)))


def parpadeo(t, t0, dur=0.14):
    return 1 - 0.93 * pulso(t, t0, dur)


def ruido(t, f, seed=0.0):
    """Temblor periodico en 1 s para que los bucles sigan cerrando."""
    return (sin(TAU * f * t + seed) + 0.5 * sin(TAU * (2 * f + 1) * t + 2.1 * seed)) / 1.5


def alas(d, abre, sube, gira=0.0, punta=0.0):
    d["wing.L"] = (gira, -sube, -abre)
    d["wing.R"] = (gira, sube, abre)
    d["wing_tip.L"] = (0, -punta, 0)
    d["wing_tip.R"] = (0, punta, 0)


def ala(d, lado, abre, sube, gira=0.0, punta=0.0):
    k = 1 if lado == "L" else -1
    d["wing." + lado] = (gira, -sube * k, -abre * k)
    d["wing_tip." + lado] = (0, -punta * k, 0)


def paso(fase, zancada, alto, apoyo=0.55):
    """Ciclo de una pata en el sitio. Devuelve (y, z, inclinacion del pie, dedos)."""
    fase %= 1.0
    if fase < apoyo:
        x = fase / apoyo
        pie = 30 * ease((x - 0.68) / 0.32)
        return (-zancada + 2 * zancada * x, 0.0, pie, -pie)
    x = (fase - apoyo) / (1 - apoyo)
    y = zancada - 2 * zancada * ease(x)
    return (y, alto * sin(math.pi * x), 30 * (1 - x) - 34 * sin(math.pi * x * 0.9) * x, 26 * sin(math.pi * x))


# ---- Pie PLANTADO (QA 0.9.3, "parece que patinan"): en Walk/Run el punto que mide el juego (cabeza del hueso toes, la
# base de los dedos) va en el apoyo hacia atrás en línea recta y a velocidad constante, plano en el suelo; el talón se
# levanta girando sobre esa base (antes giraba sobre el tobillo y los dedos resbalaban). En el balanceo vuelve por
# delante con una curva de Hermite que sale y llega a la velocidad del apoyo (sin frenazo al tocar suelo). El squash
# de la raíz (escala y bote) se descuenta en resolver(): con él los pies apoyados se movían con la raíz.
PLANTADOS = {"Walk", "Run"}


def _hermite(u, p0, p1, m0, m1):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def paso_plantado(fase, zancada, alto, apoyo=0.55, talon=30.0):
    """Base de los dedos en el sitio (y, z respecto a reposo), inclinación del pie (+ = talón arriba) y dedos."""
    fase %= 1.0
    if fase < apoyo:
        x = fase / apoyo
        pie = talon * ease((x - 0.62) / 0.38)
        return (-zancada + 2 * zancada * x, 0.0, pie, -pie)
    x = (fase - apoyo) / (1 - apoyo)
    m = 2 * zancada / apoyo * (1 - apoyo)               # velocidad del apoyo en unidades de x (hacia +Y = atrás)
    y = _hermite(x, zancada, -zancada, m, m)
    return (y, alto * sin(math.pi * x), talon * (1 - x) - 34 * sin(math.pi * x * 0.9) * x, 26 * sin(math.pi * x))


def patas_plantadas(d, fase, zancada, alto, apoyo=0.55):
    """Como patas(), pero el canal del tobillo sale de la base de los dedos plantada (en el espacio del mundo; resolver()
    lo pasa al de la raíz)."""
    for lado, f in (("L", fase), ("R", fase + 0.5)):
        y, z, pie, dedos = paso_plantado(f, zancada, alto, apoyo)
        base = Hb["toes." + lado] + Vector((0.0, y, z))
        d["base." + lado] = tuple(base)                  # resolver() pone el tobillo desde aquí, ya en la raíz
        d["foot." + lado] = pie
        d["toes." + lado] = dedos


def patas(d, fase, zancada, alto, apoyo=0.55):
    for lado, f, dx in (("L", fase, 1), ("R", fase + 0.5, -1)):
        y, z, pie, dedos = paso(f, zancada, alto, apoyo)
        d["ankle." + lado] = (0.0, y, z)
        d["foot." + lado] = pie
        d["toes." + lado] = dedos


def base_sentada(d, b):
    """Pose de empollar. b = 0 de pie, 1 sentada."""
    z = d.get("hips_l", (0, 0, 0))
    d["hips_l"] = (z[0], z[1], z[2] - 0.098 * b)
    for lado, k in (("L", 1), ("R", -1)):
        d["ankle." + lado] = (-0.012 * k * b, 0.0, 0.022 * b)
    d["neck_01"] = suma(d.get("neck_01"), (-7 * b, 0, 0))
    d["neck_02"] = suma(d.get("neck_02"), (-6 * b, 0, 0))
    d["head"] = suma(d.get("head"), (13 * b, 0, 0))


def suma(a, b):
    a = a or (0, 0, 0)
    return tuple(x + y for x, y in zip(a, b))


# ---------------------------------------------------------------- clips: base
@clip(3.0, loop=True)
def Idle(t, T):
    d = {}
    b = sin(TAU * 2 * t / T)
    d["scale"] = 1 + 0.014 * b
    d["hips_l"] = (0, 0, 0.004 * b)
    d["spine"] = (1.6 * b, 0, 0)
    # Cabezazos secos: la gallina mira a saltos, no gira suave.
    y = K(t, (0.0, 0), (0.45, 0), (0.56, 26, "o"), (1.3, 26), (1.41, -16, "o"), (2.15, -16), (2.26, 9, "o"), (2.82, 9), (3.0, 0))
    p = K(t, (0.0, 0), (1.3, 0), (1.41, 9, "o"), (2.15, 9), (2.26, -5, "o"), (2.82, -5), (3.0, 0))
    d["neck_02"] = (0.8 * b, 0, y * 0.3)
    d["head"] = (p, y * 0.34, y * 0.7)
    v = pulso(t, 1.72, 0.42)
    d["tail"] = (5 * v, 0, 15 * sin(TAU * 7 * t) * v)
    d["fan"] = 0.3 * v
    alas(d, 2 + 1.5 * b, 1 + 1.2 * b)
    d["eye"] = min(parpadeo(t, 0.86), parpadeo(t, 2.46))
    # Cambio de peso con un pasito
    d["foot.R"] = -8 * pulso(t, 1.9, 0.25)
    return d


@clip(3.0)
def Idle_Preen(t, T):
    d = {}
    g = K(t, (0, 0), (0.3, 1, "o"), (1.55, 1), (1.85, 0))
    nib = g * max(0.0, sin(TAU * 4.6 * (t - 0.4))) if 0.4 < t < 1.55 else 0.0
    d["neck_01"] = (6 * g, 0, 34 * g)
    d["neck_02"] = (10 * g, -8 * g, 40 * g)
    d["head"] = (24 * g + 13 * nib, 16 * g, 48 * g)
    d["jaw"] = 13 * nib
    d["spine"] = (0, 5 * g, 10 * g)
    d["hips"] = (0, 3 * g, 3 * g)
    ala(d, "L", 15 * g, 20 * g, 0, 10 * g)
    ala(d, "R", 2, 1)
    # Sacudida final con todas las plumas
    s = pulso(t, 2.02, 0.72)
    sac = s * sin(TAU * 7 * t)
    d["hips"] = suma(d["hips"], (0, 7 * sac, 11 * sac))
    d["spine"] = suma(d["spine"], (0, -6 * sac, -9 * sac))
    d["head"] = suma(d["head"], (0, 7 * sac, 14 * sac))
    d["tail"] = (6 * s, 0, -16 * sac)
    d["fan"] = 0.9 * s
    alas2 = 16 * s + 5 * s * sac
    ala(d, "L", 15 * g + alas2, 20 * g + alas2 * 0.7, 0, 10 * g)
    ala(d, "R", 2 + alas2, 1 + alas2 * 0.7)
    d["scale"] = 1 + 0.04 * s
    d["eye"] = 1 - 0.75 * g + 0.2 * s
    return d


@clip(0.6, loop=True)
def Walk(t, T):
    d = {}
    p = t / T
    d["hips"] = (5, 12 * sin(TAU * p + 0.25), 10 * cos(TAU * p))
    d["hips_l"] = (0.014 * sin(TAU * p), 0, 0.016 * cos(2 * TAU * p + 0.6))
    d["scale"] = 1 + 0.04 * cos(2 * TAU * p + 0.2)
    empuje = sin(2 * TAU * p) + 0.35 * sin(4 * TAU * p)     # cabeceo: adelanta y espera
    d["spine"] = (2, -3 * sin(TAU * p), -3 * cos(TAU * p))
    d["neck_01"] = (6 + 9 * empuje, -3 * sin(TAU * p), -2 * cos(TAU * p))
    d["neck_02"] = (3 + 6 * empuje, 0, 0)
    d["head"] = (-16 - 15 * empuje, -4 * sin(TAU * p), -2 * cos(TAU * p))
    alas(d, 10 + 6 * cos(2 * TAU * p), 9 + 7 * cos(2 * TAU * p + 0.7))
    d["tail"] = (5 * cos(2 * TAU * p), 0, -15 * sin(TAU * p - 0.5))
    patas_plantadas(d, p, 0.095, 0.062)          # antes 0.078 en 0.8 s (QA 0.9.3: paso más largo)
    return d


@clip(12 / FPS, loop=True)
def Run(t, T):
    d = {}
    p = t / T
    d["hips"] = (21 + 3 * cos(2 * TAU * p), 7 * sin(TAU * p), 6 * cos(TAU * p))
    d["hips_l"] = (0.012 * sin(TAU * p), 0, 0.022 * cos(2 * TAU * p))
    d["root_l"] = (0, 0, 0.022 * (1 + cos(2 * TAU * p + 2.4)))
    d["scale"] = 1 + 0.05 * cos(2 * TAU * p + 1.4)
    d["spine"] = (8, -3 * sin(TAU * p), 0)
    d["neck_01"] = (15 + 5 * sin(2 * TAU * p), 0, 0)
    d["neck_02"] = (5, 0, 0)
    d["head"] = (-48 - 5 * sin(2 * TAU * p), -5 * sin(TAU * p), -5 * cos(TAU * p))
    d["jaw"] = 13 + 6 * sin(2 * TAU * p)
    # Brazos locos: las alas van abiertas y aletean a contratiempo de las patas
    a = sin(2 * TAU * p + 0.8)
    alas(d, 52 + 10 * a, 26 + 30 * a, -8, 20 * a)
    d["tail"] = (12 + 10 * cos(2 * TAU * p), 0, -8 * sin(TAU * p))
    d["fan"] = 0.55
    d["eye"] = 1.15
    patas_plantadas(d, p, 0.12, 0.098, apoyo=0.42)   # antes 0.105 en 14 fotogramas
    return d


# ---------------------------------------------------------------- clips: sentada
@clip(0.6)
def Sit_Down(t, T):
    d = {}
    b = K(t, (0.0, 0), (0.1, -0.13), (0.26, 1, "o"))
    d["scale"] = K(t, (0, 1), (0.1, 1.06), (0.26, 0.86), (0.4, 1.03), (0.6, 0.985))
    w = pulso(t, 0.26, 0.34)
    v = sin(TAU * 7 * t) * w
    d["hips"] = (-3 * max(b, 0), 6 * v, 7 * v)
    d["tail"] = (9 * w, 0, -18 * v)
    d["fan"] = 0.6 * w
    alas(d, 5 + 14 * w, 3 + 10 * w)
    base_sentada(d, b)
    d["eye"] = K(t, (0, 1), (0.26, 0.45), (0.6, 0.72))
    return d


@clip(2.0, loop=True)
def Sit_Idle(t, T):
    d = {}
    b = sin(TAU * t / T)
    d["scale"] = 0.985 + 0.018 * b
    d["hips"] = (-3, 0, 0)
    d["spine"] = (1.5 * b, 0, 0)
    alas(d, 5 + 1.5 * b, 3 + b)
    d["head"] = (3 * sin(TAU * t / T + 1), 5 * sin(TAU * t / T), 0)
    base_sentada(d, 1)
    d["eye"] = 0.72 * parpadeo(t, 1.2, 0.3)
    d["tail"] = (0, 0, 3 * b)
    return d


@clip(14 / FPS)
def Stand_Up(t, T):
    d = {}
    b = K(t, (0.0, 1), (0.1, 1.08), (0.24, 0, "o"))
    d["scale"] = K(t, (0, 0.985), (0.1, 0.9), (0.22, 1.13), (0.36, 0.96), (0.47, 1))
    w = pulso(t, 0.2, 0.27)
    v = sin(TAU * 7.5 * t) * w
    d["hips"] = (-3 * max(0, b), 5 * v, 6 * v)
    d["tail"] = (6 * w, 0, 12 * v)
    alas(d, 5 * max(0, b) + 16 * w, 3 * max(0, b) + 14 * w)
    d["fan"] = 0.5 * w
    base_sentada(d, b)
    d["eye"] = K(t, (0, 0.72), (0.2, 1.1), (0.47, 1))
    return d


# ---------------------------------------------------------------- clips: acciones
PECK = dict(hips=(25, 0, 0), spine=(13, 0, 0), neck_01=(24, 0, 0), neck_02=(16, 0, 0), head=(6, 0, 0))


def picotazo(d, g, atras):
    for k, v in PECK.items():
        extra = -atras * {"neck_01": 12, "neck_02": 10, "head": 8}.get(k, 0)
        d[k] = (v[0] * g + extra, 0, 0)


@clip(0.4)
def Peck(t, T):
    d = {}
    atras = pulso(t, 0.0, 0.17)
    g = K(t, (0.07, 0), (0.17, 1), (0.21, 1), (0.34, -0.06), (0.4, 0))
    picotazo(d, g, atras)
    golpe = pulso(t, 0.15, 0.09)
    d["scale"] = 1 - 0.06 * golpe + 0.04 * atras
    d["jaw"] = 16 * pulso(t, 0.11, 0.1)
    d["tail"] = (15 * g, 0, 0)
    d["fan"] = 0.5 * g
    alas(d, 12 * g, 8 * g)
    d["eye"] = 1 - 0.45 * golpe
    return d


@clip(0.25)
def Eat(t, T):
    d = {}
    g = K(t, (0.0, 0.0), (0.09, 1), (0.13, 0.82), (0.17, 1), (0.25, 0))
    picotazo(d, g, 0)
    d["jaw"] = 22 * pulso(t, 0.08, 0.1)
    d["scale"] = 1 - 0.04 * pulso(t, 0.07, 0.08)
    d["tail"] = (13 * g, 0, 5 * sin(TAU * 8 * t))
    alas(d, 9 * g, 6 * g)
    d["eye"] = 1 - 0.65 * g
    return d


@clip(14 / FPS)
def Cluck(t, T):
    d = {}
    atras = K(t, (0, 0), (0.1, 1), (0.17, 0))
    fuera = K(t, (0.1, 0), (0.17, 1, "o"), (0.3, 0.75), (0.47, 0))
    d["neck_01"] = (-13 * atras + 16 * fuera, 0, 0)
    d["neck_02"] = (-11 * atras + 13 * fuera, 0, 0)
    d["head"] = (6 * atras - 24 * fuera, 0, 8 * sin(TAU * 7 * t) * fuera)
    d["hips"] = (-5 * atras + 7 * fuera, 0, 0)
    d["jaw"] = 34 * K(t, (0.11, 0), (0.17, 1, "o"), (0.31, 0.75), (0.42, 0))
    d["scale"] = 1 + 0.07 * atras - 0.08 * fuera
    d["root_l"] = (0, 0, 0.035 * pulso(t, 0.13, 0.17))
    d["tail"] = (17 * fuera, 0, 0)
    d["fan"] = 0.65 * fuera
    alas(d, 13 * fuera, 12 * fuera)
    d["eye"] = 1 - 0.4 * atras + 0.2 * fuera
    return d


@clip(1.3)
def Look_Around(t, T):
    d = {}
    y = K(t, (0.08, 0), (0.17, 62, "o"), (0.58, 62), (0.67, -62, "o"), (1.1, -62), (1.22, 0, "o"))
    d["neck_02"] = (0, 0, y * 0.32)
    d["head"] = (K(t, (0.08, 0), (0.17, -8), (0.58, -8), (0.67, 6), (1.1, 6), (1.22, 0)), y * 0.2, y * 0.68)
    d["spine"] = (0, 0, y * 0.1)
    d["hips_l"] = (0, 0, 0.012 * pulso(t, 0.0, 1.3))
    d["scale"] = 1 + 0.03 * pulso(t, 0.0, 0.2) + 0.03 * pulso(t, 0.56, 0.16)
    d["eye"] = min(1.18, 1.18 * parpadeo(t, 0.6, 0.1))
    alas(d, 3, 2)
    return d


@clip(0.6)
def Flap(t, T):
    d = {}
    env = K(t, (0, 0), (0.12, 1, "o"), (0.42, 0.9), (0.6, 0))
    bate = sin(TAU * 5 * t)
    alas(d, 76 * env, env * (20 + 46 * bate), -6 * env, 24 * bate * env)
    d["scale"] = 1 + 0.13 * pulso(t, 0.02, 0.2) - 0.08 * pulso(t, 0.4, 0.2)
    d["root_l"] = (0, 0, 0.07 * pulso(t, 0.03, 0.46))
    d["hips"] = (-14 * env, 0, 0)
    d["neck_01"] = (5 * env, 0, 0)
    d["head"] = (9 * env, 0, 0)
    d["tail"] = (-8 * env + 8 * bate * env, 0, 0)
    d["fan"] = 1.0 * env
    d["jaw"] = 16 * env
    d["eye"] = 1 + 0.15 * env
    for lado, k in (("L", 1), ("R", -1)):
        d["ankle." + lado] = (0.012 * k * env, 0.012 * env, 0.028 * env * 0.5)
        d["foot." + lado] = 30 * env
    return d


@clip(1.0)
def Panic(t, T):
    d = {}
    env = K(t, (0, 0), (0.12, 1, "o"), (0.85, 1), (1.0, 0))
    bate = sin(TAU * 6 * t)
    alas(d, 70 * env, env * (24 + 42 * bate), -8 * env, 22 * bate * env)
    d["root_l"] = (0, 0, env * 0.05 * abs(sin(TAU * 3 * t)))
    d["scale"] = 1 + env * 0.09 * cos(TAU * 6 * t)
    d["hips"] = (-5 * env, 7 * env * sin(TAU * 5 * t), 13 * env * sin(TAU * 4 * t))
    d["neck_01"] = (-8 * env, 0, 16 * env * sin(TAU * 5 * t + 1))
    d["head"] = (6 * env, 16 * env * sin(TAU * 6 * t), 34 * env * sin(TAU * 5 * t))
    d["jaw"] = env * (26 + 12 * sin(TAU * 7 * t))
    d["tail"] = (12 * env, 0, 15 * env * sin(TAU * 6 * t))
    d["fan"] = env
    d["eye"] = 1 + 0.42 * env
    d2 = {}
    patas(d2, t * 4, 0.075, 0.09, apoyo=0.45)
    for k, v in d2.items():
        d[k] = tuple(x * env for x in v) if isinstance(v, tuple) else v * env
    return d


# ---------------------------------------------------------------- clips: aire
@clip(7 / FPS)
def Jump(t, T):
    d = {}
    c = K(t, (0, 0), (0.1, 1), (0.233, -1))    # agacha y se estira
    d["scale"] = 1 - 0.2 * c
    d["hips_l"] = (0, 0, -0.05 * max(0, c))
    d["hips"] = (9 * max(0, c) - 13 * max(0, -c), 0, 0)
    d["head"] = (8 * max(0, c), 0, 0)
    d["neck_01"] = (8 * max(0, c) - 5 * max(0, -c), 0, 0)
    alas(d, 22 * max(0, c) + 72 * max(0, -c), -14 * max(0, c) + 56 * max(0, -c) * (1 + c) - 15 * max(0, -c), 0, 15 * max(0, -c))
    d["tail"] = (6 * max(0, c) - 10 * max(0, -c), 0, 0)
    d["fan"] = 0.8 * max(0, -c)
    d["eye"] = 1 - 0.4 * max(0, c) + 0.15 * max(0, -c)
    for lado in "LR":
        d["foot." + lado] = 34 * max(0, -c)
        d["toes." + lado] = 30 * max(0, -c)
    return d


@clip(8 / FPS, loop=True)
def Air(t, T):
    d = {}
    p = t / T
    b = sin(TAU * p)
    alas(d, 74, 18 + 48 * b, -6, 26 * sin(TAU * p - 0.8))
    d["scale"] = 1.05 + 0.035 * b
    d["hips"] = (-14 + 4 * b, 0, 0)
    d["neck_01"] = (4, 0, 0)
    d["head"] = (10 - 3 * b, 0, 0)
    d["jaw"] = 12
    d["tail"] = (-10 + 10 * sin(TAU * p + 1), 0, 0)
    d["fan"] = 0.85
    d["eye"] = 1.12
    for lado, f, k in (("L", 0.0, 1), ("R", 0.5, -1)):
        d["ankle." + lado] = (0.014 * k, 0.03 * sin(TAU * (p + f)), 0.03)
        d["foot." + lado] = 42 + 12 * sin(TAU * (p + f))
        d["toes." + lado] = 30
    return d


@clip(0.4, loop=True)
def Fall(t, T):
    d = {}
    p = t / T
    b = sin(TAU * 2 * p)
    alas(d, 78, 58 + 10 * b, -4, -18 + 10 * b)
    d["scale"] = 1.09 + 0.015 * b
    d["hips"] = (-6, 3 * sin(TAU * p), 0)
    d["neck_01"] = (-5, 0, 0)
    d["head"] = (3, 4 * sin(TAU * p), 0)
    d["jaw"] = 20 + 6 * b
    d["tail"] = (16, 0, 5 * sin(TAU * p))
    d["fan"] = 1.0
    d["eye"] = 1.3
    for lado, f, k in (("L", 0.0, 1), ("R", 0.5, -1)):
        d["ankle." + lado] = (0.016 * k, 0.022 * sin(TAU * (2 * p + f)), -0.016)
        d["foot." + lado] = 38
        d["toes." + lado] = 20 + 14 * sin(TAU * (2 * p + f))
    return d


@clip(0.5)
def Land(t, T):
    d = {}
    amort = math.exp(-7.5 * t) * cos(TAU * 3.6 * t)
    d["scale"] = 1 - 0.27 * amort
    d["hips_l"] = (0, 0, -0.07 * max(0.0, amort))
    d["hips"] = (9 * amort, 0, 0)
    d["neck_01"] = (11 * amort, 0, 0)
    d["neck_02"] = (7 * amort, 0, 0)
    d["head"] = (-20 * amort, 0, 0)
    # Las alas llegan abiertas y se recogen con un sacudon
    g = K(t, (0, 1), (0.28, 0))
    alas(d, 74 * g, 50 * g - 18 * pulso(t, 0.0, 0.22), 0, -14 * g)
    d["tail"] = (-14 * amort + 8 * g, 0, 0)
    d["fan"] = 0.9 * g
    d["eye"] = 0.3 + 0.7 * ease(t / 0.2)
    d["jaw"] = 12 * g
    return d


# ---------------------------------------------------------------- clips: estados
@clip(1.0, loop=True)
def Netted(t, T):
    d = {}
    tiron = max(0.0, sin(TAU * 2 * t)) ** 3
    lado = sin(TAU * t)
    d["hips_l"] = (0.02 * lado, 0, -0.05 - 0.01 * tiron)
    d["hips"] = (9, 8 * lado, 13 * lado * (0.5 + tiron))
    d["scale"] = 1 + 0.06 * tiron
    d["spine"] = (2, -5 * lado, -7 * lado)
    d["neck_01"] = (8 * tiron, 8 * lado, 13 * sin(TAU * 2 * t + 1))
    d["neck_02"] = (8, 0, 14 * sin(TAU * 3 * t))
    d["head"] = (-20 - 6 * tiron, 12 * sin(TAU * 3 * t), 24 * sin(TAU * 2 * t))
    d["jaw"] = 16 + 12 * tiron
    a = sin(TAU * 6 * t)
    alas(d, 13 + 8 * tiron, 8 + 8 * a + 6 * tiron, 0, 10 * a)
    d["tail"] = (9 * tiron, 0, -14 * lado)
    d["fan"] = 0.4 + 0.5 * tiron
    d["eye"] = 1.3
    d2 = {}
    patas(d2, t * 3, 0.035, 0.03, apoyo=0.5)
    d.update(d2)
    return d


@clip(16 / FPS, loop=True)
def Carried(t, T):
    d = {}
    p = t / T
    d["hips"] = (-22 + 4 * sin(2 * TAU * p), 9 * sin(TAU * p), 11 * sin(TAU * p + 1))
    d["neck_01"] = (10, 0, 12 * sin(TAU * p + 2))
    d["neck_02"] = (6, 0, 12 * sin(2 * TAU * p))
    d["head"] = (15, 12 * sin(TAU * p), 26 * sin(2 * TAU * p + 1))
    d["jaw"] = 22 + 13 * sin(3 * TAU * p)
    a = sin(3 * TAU * p)
    alas(d, 66, 24 + 44 * a, -8, 24 * a)
    d["scale"] = 1.04 + 0.04 * sin(3 * TAU * p + 1)
    d["tail"] = (9, 0, 15 * sin(TAU * p))
    d["fan"] = 0.9
    d["eye"] = 1.38
    for lado, f, k in (("L", 0.0, 1), ("R", 0.5, -1)):
        x = sin(TAU * (2 * p + f))
        d["ankle." + lado] = (0.012 * k, 0.066 * x, 0.022 + 0.022 * cos(TAU * (2 * p + f)))
        d["foot." + lado] = 22 + 28 * x
        d["toes." + lado] = 16 - 22 * x
    return d


@clip(1.6, loop=True)
def Stunned(t, T):
    d = {}
    a = TAU * t / T
    d["hips"] = (5, 5 * sin(a), 3 * cos(a))
    d["spine"] = (2, 3 * sin(a - 0.5), 0)
    d["neck_01"] = (5 + 5 * cos(a), 7 * sin(a), 0)
    d["neck_02"] = (7 * cos(a - 0.4), 10 * sin(a - 0.4), 0)
    d["head"] = (-8 + 11 * cos(a - 0.8), 16 * sin(a - 0.8), 8 * cos(a - 0.8))
    d["jaw"] = 15 + 3 * sin(a * 2)
    alas(d, 19, -7 + 3 * sin(a * 2), 0, -12)
    d["tail"] = (-13, 0, 5 * sin(a))
    d["scale"] = 0.97 + 0.012 * sin(2 * a)
    base_sentada(d, 0.82)
    d["eye"] = 0.32
    return d


@clip(0.8, loop=True)
def Victory(t, T):
    d = {}
    p = (t / T * 2) % 1.0        # dos saltitos por ciclo
    sube = 4 * p * (1 - p)
    d["root_l"] = (0, 0, 0.17 * sube)
    d["scale"] = 1 + 0.15 * (sube - 0.55) * (1 if sube > 0.55 else 1.7)
    giro = sin(TAU * t / T)
    d["hips"] = (-9 * sube, 7 * giro, 15 * giro)
    d["spine"] = (0, -4 * giro, -5 * giro)
    d["neck_01"] = (-5, 0, 0)
    d["head"] = (-11 * sube + 4, 13 * giro, 9 * giro)
    d["jaw"] = 12 + 22 * sube
    a = sin(TAU * 3 * t / T * 2)
    alas(d, 66, 38 + 26 * a, -8, 18 * a)
    d["tail"] = (14 * sube, 0, -13 * giro)
    d["fan"] = 0.55 + 0.45 * sube
    d["eye"] = 0.68
    for lado, k in (("L", 1), ("R", -1)):
        d["ankle." + lado] = (0.008 * k * sube, 0, 0.042 * sube)
        d["foot." + lado] = 28 * sube
        d["toes." + lado] = 26 * sube
    return d


# ---------------------------------------------------------------- gestos (rueda de gestos del juego)
@clip(1.6, loop=True)
def Dance(t, T):
    """El baile del pollito de verdad: aletea a tiempo (dos golpes de ala por compas), menea la cadera, pasito a
    pasito y la cabeza va y viene como un metronomo. Cuatro tiempos por ciclo."""
    d = {}
    u = t / T
    beat = (u * 4) % 1.0
    golpe = back(min(1.0, beat * 3.0)) * (1 - ease((beat - 0.35) / 0.65))          # sube de golpe y baja suave
    rebote = abs(sin(TAU * u * 2))
    meneo = sin(TAU * u)
    d["root_l"] = (0, 0, 0.018 * rebote)
    d["scale"] = 1 + 0.03 * (rebote - 0.5)
    d["hips"] = (-4 * rebote, 6 * meneo, 16 * meneo)
    d["spine"] = (3 * golpe, -3 * meneo, -6 * meneo)
    cab = K(beat, (0, 0), (0.15, 1, "o"), (0.5, -0.6), (1, 0))                    # picoteo al compas
    d["neck_01"] = (10 * cab, 0, -8 * meneo)
    d["neck_02"] = (8 * cab, 0, -10 * meneo)
    d["head"] = (-14 * cab, 8 * meneo, -14 * meneo)
    d["jaw"] = 10 * pulso(beat, 0.1, 0.3)
    alas(d, 30 + 42 * golpe, 18 + 40 * golpe, -6, 22 * golpe)
    d["tail"] = (8 + 6 * rebote, 0, -18 * meneo)
    d["fan"] = 0.35 + 0.4 * golpe
    patas(d, u * 2, 0.012, 0.03, apoyo=0.6)
    d["eye"] = min(parpadeo(t, 0.55 * T), 0.95)
    return d


@clip(1.2, loop=True)
def Wave(t, T):
    """¡Hola!: levanta el ala izquierda bien alta y la agita (la punta va con retraso), ladea la cabeza y da un
    saltito de alegria al principio de cada vaiven."""
    d = {}
    u = t / T
    w = sin(TAU * u * 2)
    wp = sin(TAU * u * 2 - 0.7)
    salto = pulso(u, 0.0, 0.25) + pulso(u, 0.5, 0.25)
    d["root_l"] = (0, 0, 0.03 * salto)
    d["hips"] = (-3, -5, 6)
    d["spine"] = (0, -5, 4)
    d["neck_02"] = (-4, 0, 8)
    d["head"] = (-6, 18 + 4 * w, 14)
    d["jaw"] = 8 * salto
    ala(d, "L", 38 + 12 * w, 126 + 18 * w, -18, 34 * wp)     # bien arriba (sube) y algo abierta, la punta con retraso
    ala(d, "R", 4, 3)
    d["tail"] = (6, 0, 8 * w)
    d["fan"] = 0.4
    for lado in ("L", "R"):
        d["ankle." + lado] = (0, 0, 0.02 * salto)
        d["foot." + lado] = 16 * salto
    d["eye"] = 0.8
    return d


@clip(1.2, loop=True)
def Taunt(t, T):
    """Burla de gallina: se agacha de frente, levanta la cola y la menea muy rapido abriendo el abanico, mira por
    encima del hombro y cacarea burlona."""
    d = {}
    u = t / T
    meneo = sin(TAU * u * 6)
    risa = abs(sin(TAU * u * 4))
    d["hips"] = (18, 0, 6 * meneo)
    d["spine"] = (6, 0, -4 * meneo)
    d["root_l"] = (0, 0, -0.01)
    d["neck_01"] = (-10, 0, 22)
    d["neck_02"] = (-8, 0, 30)
    d["head"] = (4, -12, 44 + 6 * sin(TAU * u * 2))
    d["jaw"] = 26 * risa
    d["tail"] = (-28, 0, 34 * meneo)
    d["fan"] = 1.0
    alas(d, 22 + 8 * risa, 14 + 10 * risa, 0, 10 * meneo)
    d["scale"] = 1 + 0.02 * risa
    d["eye"] = 0.6
    return d


# ---------------------------------------------------------------- presentacion del juego
@clip(4.2)
def Intro(t, T):
    """Cae del cielo, aterriza aplastada, mira a los lados, coge aire y suelta un cacareo enorme (entra el logo) y
    termina con dos saltitos de orgullo. El juego la mueve y pone el logo encima del cacareo (t = 2.05 s)."""
    d = {}
    # 1) Caida (0 - 0.42): aleteo loco, patas colgando, ojos como platos.
    if t < 0.42:
        x = t / 0.42
        d["root_l"] = (0, 0, 1.25 * (1 - x * x))
        b = sin(TAU * 7 * t)
        alas(d, 76, 20 + 46 * b, -6, 24 * b)
        d["scale"] = 1.08 + 0.04 * b
        d["hips"] = (-12, 0, 0)
        d["head"] = (12, 0, 0)
        d["jaw"] = 22
        d["tail"] = (16, 0, 0)
        d["fan"] = 1.0
        d["eye"] = 1.38
        for lado, f, k in (("L", 0.0, 1), ("R", 0.5, -1)):
            d["ankle." + lado] = (0.016 * k, 0.03 * sin(TAU * 3 * t + f * TAU), -0.016)
            d["foot." + lado] = 40
            d["toes." + lado] = 24 + 12 * sin(TAU * 4 * t + f * TAU)
        return d

    # 2) Aterrizaje (0.42 - 0.95): aplastada con rebote, alas que se recogen de un sacudon.
    u = t - 0.42
    amort = math.exp(-6.5 * u) * cos(TAU * 3.4 * u)
    caida = 1.0 if t < 0.95 else 0.0
    d["scale"] = 1 - 0.32 * amort * caida
    d["hips_l"] = (0, 0, -0.08 * max(0.0, amort) * caida)
    d["hips"] = (9 * amort * caida, 0, 0)
    d["neck_01"] = (11 * amort * caida, 0, 0)
    d["head"] = (-20 * amort * caida, 0, 0)
    g = K(t, (0.42, 1), (0.72, 0))
    alas(d, 74 * g, 52 * g, 0, -14 * g)
    d["tail"] = (-12 * amort * caida + 10 * g, 0, 0)
    d["fan"] = 0.9 * g
    d["eye"] = K(t, (0.42, 0.25), (0.62, 1.0))

    # 3) Mira a un lado y a otro (0.95 - 1.6), cabezazos secos.
    y = K(t, (0.95, 0), (1.03, 58, "o"), (1.24, 58), (1.32, -58, "o"), (1.52, -58), (1.6, 0, "o"))
    d["neck_02"] = (0, 0, y * 0.32)
    d["head"] = suma(d["head"], (0, y * 0.2, y * 0.68))
    if 0.95 <= t < 1.6:
        d["eye"] = min(1.15, 1.15 * parpadeo(t, 1.25, 0.1))

    # 4) Coge aire (1.6 - 2.02): se agacha hacia atras, se infla, alas recogidas temblando.
    aire = K(t, (1.6, 0), (2.0, 1), (2.06, 0))
    # 5) Cacareo enorme (2.02 - 2.85): se estira hacia la camara, pico muy abierto, alas y cola en abanico.
    grito = K(t, (2.0, 0), (2.08, 1, "o"), (2.62, 0.92), (2.85, 0))
    tiembla = sin(TAU * 11 * t)
    d["scale"] = d["scale"] * (1 + 0.1 * aire + 0.12 * grito * (1 + 0.15 * tiembla)) - 0.06 * aire
    d["hips"] = suma(d["hips"], (-7 * aire + 10 * grito, 0, 0))
    d["spine"] = (-6 * aire + 8 * grito, 0, 0)
    d["neck_01"] = suma(d["neck_01"], (-16 * aire + 18 * grito, 0, 0))
    d["neck_02"] = suma(d["neck_02"], (-12 * aire + 14 * grito, 0, 4 * grito * tiembla))
    d["head"] = suma(d["head"], (10 * aire - 34 * grito, 4 * grito * tiembla, 6 * grito * tiembla))
    d["jaw"] = 42 * grito + 6 * aire
    d["root_l"] = (0, 0, 0.06 * pulso(t, 2.0, 0.3))
    if aire > 0.01 or grito > 0.01:
        bate = sin(TAU * 6 * t)
        alas(d, 14 * aire + 80 * grito, 6 * aire + grito * (30 + 40 * bate), -8 * grito, 22 * bate * grito)
        d["tail"] = (-6 * aire + 22 * grito, 0, 10 * grito * tiembla)
        d["fan"] = 0.3 * aire + 1.0 * grito
        d["eye"] = 1 - 0.5 * aire + 0.35 * grito

    # 6) Orgullo (2.85 - 4.2): dos saltitos de alegria, ojos felices, y se queda en pose.
    if t >= 2.85:
        p = min(1.0, (t - 2.9) / 0.7) if t > 2.9 else 0.0
        q = (p * 2) % 1.0 if p < 1.0 else 0.0
        sube = 4 * q * (1 - q) if 0 < p < 1 else 0.0
        fin = K(t, (3.6, 0), (3.9, 1, "o"))
        d["root_l"] = (0, 0, 0.14 * sube)
        d["scale"] = 1 + 0.13 * (sube - 0.5) * (1 if sube > 0.5 else 1.6) * (1 - fin) + 0.03 * fin
        giro = sin(TAU * (t - 2.9) / 0.7)
        d["hips"] = (-8 * sube - 6 * fin, 6 * giro * (1 - fin), 12 * giro * (1 - fin))
        d["spine"] = (-3 * fin, 0, 0)
        d["neck_01"] = (-5 - 4 * fin, 0, 0)
        d["neck_02"] = (0, 0, 0)
        d["head"] = (-10 * sube + 4 - 6 * fin, 10 * giro * (1 - fin) + 8 * fin, 8 * giro * (1 - fin) + 10 * fin)
        d["jaw"] = 10 + 20 * sube
        a = sin(TAU * 6 * t)
        abre = 64 * (1 - fin) + 18 * fin
        alas(d, abre, (34 + 24 * a) * (1 - fin) + 10 * fin, -8 * (1 - fin), 16 * a * (1 - fin))
        d["tail"] = (12 * sube + 10 * fin, 0, -12 * giro * (1 - fin))
        d["fan"] = 0.5 + 0.4 * sube * (1 - fin) + 0.2 * fin
        d["eye"] = 0.62 * parpadeo(t, 4.0, 0.12) if t > 3.9 else 0.62
        for lado, k in (("L", 1), ("R", -1)):
            d["ankle." + lado] = (0.008 * k * sube, 0, 0.04 * sube)
            d["foot." + lado] = 26 * sube
            d["toes." + lado] = 24 * sube
    return d


LANZADOR_PASOS = (0.3, 0.9, 1.5, 2.1)   # pisotones (tiembla la pantalla); el launcher los lee de render_launcher.py


@clip(5.2)
def Launcher_Intro(t, T):
    """Gallina GIGANTE para el launcher (render_launcher.py la pinta sobre el escritorio): cruza de perfil con
    pisotones pesados, se para con un saltito girándose hacia ti, se agacha, ladea la cabeza con curiosidad y abre el
    pico enorme acercándose, como si fuera a tragarte (de ahí sale la ventana del launcher)."""
    d = {}
    # 1) Andar de gigante (0 - 2.4): ciclo lento de 1.2 s, pasos largos, cuerpo que se hunde en cada pisotón.
    if t < 2.4:
        p = (t / 1.2) % 1.0
        d["root"] = (0, 0, 90)                                   # de perfil, mirando a la derecha de la pantalla
        golpe = max(0.0, cos(2 * TAU * p + 0.35)) ** 6           # aplastón justo al apoyar
        d["hips"] = (8 + 4 * cos(2 * TAU * p), 9 * sin(TAU * p), 9 * cos(TAU * p))
        d["hips_l"] = (0.016 * sin(TAU * p), 0, 0.03 * cos(2 * TAU * p) - 0.02 * golpe)
        d["scale"] = 1 + 0.05 * cos(2 * TAU * p + 0.2) - 0.07 * golpe
        empuje = sin(2 * TAU * p) + 0.35 * sin(4 * TAU * p)
        d["spine"] = (3, -4 * sin(TAU * p), -4 * cos(TAU * p))
        d["neck_01"] = (8 + 10 * empuje, -4 * sin(TAU * p), 0)
        d["neck_02"] = (4 + 7 * empuje, 0, 0)
        d["head"] = (-18 - 16 * empuje, -5 * sin(TAU * p), 0)
        alas(d, 14 + 8 * cos(2 * TAU * p), 12 + 10 * cos(2 * TAU * p + 0.7))
        d["tail"] = (6 * cos(2 * TAU * p), 0, -18 * sin(TAU * p - 0.5))
        d["eye"] = 1.0 if (t % 1.7) > 0.12 else 0.2
        patas(d, p, 0.12, 0.11, apoyo=0.5)
        return d

    # 2) Se para y se gira hacia la cámara con un saltito (2.4 - 2.95).
    giro = K(t, (2.4, 90), (2.75, -8, "o"), (2.95, 0))
    salto = pulso(t, 2.45, 0.42)
    d["root"] = (0, 0, giro)
    d["root_l"] = (0, 0, 0.09 * salto)
    aterriza = math.exp(-9 * max(0.0, t - 2.87)) * cos(TAU * 3 * max(0.0, t - 2.87)) if t > 2.87 else 0.0
    # 3) Se agacha (2.95 - 3.35) y se queda mirando: postura baja, cuello estirado hacia la cámara.
    baja = K(t, (2.95, 0), (3.3, 1, "o"))
    # 5) Abre el pico enorme y se acerca (4.3 - 5.2).
    come = K(t, (4.3, 0), (4.85, 1, "o"))
    d["scale"] = 1 + 0.08 * salto - 0.22 * aterriza - 0.08 * baja + 0.06 * come
    d["hips"] = (14 * baja + 12 * come, 0, 0)
    d["hips_l"] = (0, 0, -0.07 * baja - 0.04 * max(0.0, aterriza))
    d["spine"] = (8 * baja + 6 * come, 0, 0)
    d["neck_01"] = (-6 * baja + 22 * come, 0, 0)
    d["neck_02"] = (-8 * baja + 14 * come, 0, 0)
    # 4) Cabeceos de curiosidad (3.35 - 4.3): ladea la cabeza a un lado y a otro, secos, como las gallinas.
    lado = K(t, (3.35, 0), (3.45, 26, "o"), (3.75, 26), (3.85, -24, "o"), (4.1, -24), (4.2, 8, "o"), (4.3, 0))
    gira = K(t, (3.35, 0), (3.45, 14, "o"), (3.75, 14), (3.85, -12, "o"), (4.1, -12), (4.3, 0))
    d["head"] = (-24 * baja - 30 * come - 4 * salto, lado * (1 - come), gira * (1 - come))
    d["jaw"] = 62 * come + 8 * salto
    alas(d, 12 + 30 * salto + 10 * come, 8 + 34 * salto, 0, 12 * salto)
    d["tail"] = (12 * salto + 8 * baja + 14 * come, 0, 6 * sin(TAU * 3 * t) * baja)
    d["fan"] = 0.4 * salto + 0.8 * come
    d["eye"] = 1.15 + 0.3 * come
    if 3.95 < t < 4.1:
        d["eye"] = parpadeo(t, 3.96, 0.13) * 1.15
    for lado_p, k in (("L", 1), ("R", -1)):
        d["ankle." + lado_p] = (0.02 * k * baja, 0, 0.03 * salto)
        d["foot." + lado_p] = 20 * salto
    return d


# ---------------------------------------------------------------- muelles (movimiento secundario)
GESTOS = {"Peck", "Eat", "Cluck", "Look_Around", "Flap", "Panic", "Idle_Preen", "Land", "Stand_Up"}
AEREOS = {"Air", "Fall", "Carried"}


def muelle(senal, hz, zeta, bucle, pasadas=4):
    """Devuelve la senal filtrada por un muelle: lo que llega tarde y rebota."""
    n = len(senal)
    w = TAU * hz
    x, v = senal[0], 0.0
    sal = np.zeros(n)
    dt = 1.0 / FPS / 4
    m = n - 1 if bucle else n
    for ciclo in range(pasadas if bucle else 1):
        for i in range(m):
            for _ in range(4):
                v += (w * w * (senal[i] - x) - 2 * zeta * w * v) * dt
                x += v * dt
            sal[i] = x
    if bucle:
        sal[n - 1] = sal[0]
    return sal


def secundario(c, bucle, nombre):
    """Cresta, barbillas, puntas de ala y plumas siguen al cuerpo con retraso."""
    n = len(c["hips"])
    g = np.ones(n)
    if nombre in GESTOS:
        k = min(7, n - 1)
        g[-k:] = np.linspace(1, 0, k)

    def acum(indice, bones):
        return np.sum([c[b][:, indice] for b in bones], axis=0)

    col = ["hips", "spine", "neck_01", "neck_02", "head"]
    pitch, roll = acum(0, col), acum(1, col)
    alto = c["hips_l"][:, 2] + c["root_l"][:, 2]
    lag_p = muelle(pitch + 300 * alto, 5.2, 0.16, bucle) - (pitch + 300 * alto)
    lag_r = muelle(roll, 5.2, 0.16, bucle) - roll
    c["comb"][:, 0] += g * np.clip(lag_p * 0.9, -38, 38)
    c["comb"][:, 1] += g * np.clip(lag_r * 0.9, -30, 30)
    lag_p = muelle(pitch + 420 * alto, 4.3, 0.13, bucle) - (pitch + 420 * alto)
    lag_r = muelle(roll, 4.3, 0.13, bucle) - roll
    for s in (".L", ".R"):
        c["wattle" + s][:, 0] += g * np.clip(lag_p * 1.0, -50, 50)
        c["wattle" + s][:, 1] += g * np.clip(lag_r * 1.0, -35, 35)

    # Cola: la base sigue a la cadera, cada pluma con su propio ritmo.
    p, y = c["hips"][:, 0] + c["tail"][:, 0], c["hips"][:, 2] + c["tail"][:, 2]
    p = p - 260 * alto
    f = g
    for hueso, hz, ga in (("tail_mid", 4.6, 0.8), ("tail_high.L", 3.9, 1.0), ("tail_high.R", 4.2, 1.0),
                         ("tail_low.L", 3.4, 1.1), ("tail_low.R", 3.6, 1.1)):
        c[hueso][:, 0] += f * np.clip((muelle(p, hz, 0.16, bucle) - p) * ga, -32, 32)
        c[hueso][:, 2] += f * np.clip((muelle(y, hz, 0.16, bucle) - y) * ga, -32, 32)

    # Las puntas de las alas llegan tarde al aleteo.
    for s, k in ((".L", -1), (".R", 1)):
        sube = c["wing" + s][:, 1]
        c["wing_tip" + s][:, 1] += g * np.clip((muelle(sube, 5.8, 0.25, bucle) - sube) * 0.75, -45, 45)


# ---------------------------------------------------------------- de canales a Actions
scene = bpy.context.scene
scene.render.fps = FPS
arm = bpy.data.objects["Gallina_Rig"]
bpy.context.view_layer.objects.active = arm
data = arm.data
Rb = {b.name: b.matrix_local.to_quaternion() for b in data.bones}
Hb = {b.name: b.head_local.copy() for b in data.bones}
Tb = {b.name: b.tail_local.copy() for b in data.bones}
ROT = list(data.bones.keys())


def q_rig(e):
    return Euler((math.radians(e[0]), math.radians(e[1]), math.radians(e[2])), "XYZ").to_quaternion()


def a_local(nombre, q):
    return Rb[nombre].inverted() @ q @ Rb[nombre]


def resolver(c, i):
    """Canales de un fotograma -> rotacion/posicion/escala de cada hueso."""
    pose = {}
    for b in ROT:
        pose[b] = [a_local(b, q_rig(c[b][i])), Vector((0, 0, 0)), Vector((1, 1, 1))]

    q_h = q_rig(c["hips"][i])
    d_h = Vector(c["hips_l"][i])
    pose["hips"][1] = Rb["hips"].inverted() @ d_h
    pose["root"][1] = Rb["root"].inverted() @ Vector(c["root_l"][i])
    s = max(0.3, float(c["scale"][i]))
    pose["root"][2] = Vector((1 / math.sqrt(s), s, 1 / math.sqrt(s)))

    # Patas con IK: el tobillo va donde se le pide aunque la cadera se mueva.
    for lado in ("L", "R"):
        leg, foot, toes = "leg." + lado, "foot." + lado, "toes." + lado
        pivote = Hb["hips"]
        cabeza = d_h + pivote + q_h @ (Hb[leg] - pivote)
        tobillo = Hb[foot] + Vector(c["ankle." + lado][i])
        if c.get("_nombre") in PLANTADOS:
            # La base de los dedos pedida está en el mundo; la raíz se mueve (root_l) y se aplasta (vertical s,
            # horizontal 1/√s): se pasa a la raíz y el tobillo sale de ahí con el giro del pie.
            r_, b_ = Vector(c["root_l"][i]), Vector(c["base." + lado][i])
            b_ = Vector(((b_.x - r_.x) * math.sqrt(s), (b_.y - r_.y) * math.sqrt(s), (b_.z - r_.z) / s))
            tobillo = b_ - q_rig((c["foot_rx." + lado][i], 0, 0)) @ (Hb[toes] - Hb[foot])
        reposo = Hb[foot] - Hb[leg]
        v = tobillo - cabeza
        q_w = reposo.rotation_difference(v)
        delta = tobillo - (cabeza + q_w @ reposo)
        q_l = q_h.inverted() @ q_w
        pose[leg][0] = a_local(leg, q_l)
        pose[leg][1] = Rb[leg].inverted() @ (q_h.inverted() @ delta)
        q_pie = q_rig((c["foot_rx." + lado][i], 0, 0))
        pose[foot][0] = a_local(foot, q_w.inverted() @ q_pie)
        pose[toes][0] = a_local(toes, q_rig((c["toes_rx." + lado][i], 0, 0)))

    # Papada: se desplaza con el pico de abajo (canal "papada", en ejes del rig).
    if "papada" in c:
        for w in ("wattle.L", "wattle.R"):
            pose[w][1] = Rb[w].inverted() @ Vector(c["papada"][i])

    e = float(c["eye"][i])
    ancho = 1 + (e - 1) * 0.6 if e > 1 else 1.0
    for lado in ("L", "R"):
        pose["eye." + lado][2] = Vector((ancho, ancho, max(0.06, e)))
    return pose


def canales(fn, n, loop):
    T = n / FPS
    frames = n + 1
    c = {b: np.zeros((frames, 3)) for b in ROT}
    for k in ("hips_l", "root_l", "ankle.L", "ankle.R", "base.L", "base.R"):
        c[k] = np.zeros((frames, 3))
    for k in ("foot_rx.L", "foot_rx.R", "toes_rx.L", "toes_rx.R", "fan"):
        c[k] = np.zeros(frames)
    c["scale"], c["eye"] = np.ones(frames), np.ones(frames)
    c["_nombre"] = fn.__name__
    for i in range(frames):
        t = i / FPS
        d = fn(0.0 if (loop and i == n) else t, T)
        for k, v in d.items():
            if k in ("foot.L", "foot.R"):
                c["foot_rx." + k[-1]][i] = v
            elif k in ("toes.L", "toes.R"):
                c["toes_rx." + k[-1]][i] = v
            elif k == "jaw":
                c["jaw"][i, 0] = v
            elif k in ("fan", "scale", "eye"):
                c[k][i] = v
            else:
                c[k][i] = v
    # La papada cuelga de la cabeza, pero va bajo el pico de abajo: al abrir el pico baja y se adelanta con él (si
    # no, el pico de abajo baja por delante de ella y se ve la papada dentro de la boca). Sólo un poco de giro: con
    # el giro entero de la mandíbula se metería en el pecho.
    abre = np.sin(np.radians(np.clip(c["jaw"][:, 0], 0, 90)))
    c["papada"] = np.zeros((frames, 3))
    c["papada"][:, 1] = -0.035 * abre      # hacia delante (la gallina mira a -Y)
    c["papada"][:, 2] = -0.08 * abre       # hacia abajo
    for s in (".L", ".R"):
        c["wattle" + s][:, 0] += 0.3 * c["jaw"][:, 0]
    # Abrir el abanico: las plumas laterales se separan hacia fuera.
    for hueso, ang in (("tail_high.L", 24), ("tail_high.R", -24), ("tail_low.L", 34), ("tail_low.R", -34)):
        c[hueso][:, 1] += ang * c["fan"]
    if fn.__name__ not in AEREOS and fn.__name__ not in PLANTADOS:    # los plantados ya llevan el tobillo bien puesto
        for lado in ("L", "R"):
            a = np.radians(c["foot_rx." + lado])
            d_t = np.radians(c["toes_rx." + lado])
            baja = np.maximum.reduce([0.09 * np.sin(a) + 0.10 * np.sin(a + d_t), 0.09 * np.sin(-a), np.zeros(frames)])
            c["ankle." + lado][:, 2] = np.maximum(c["ankle." + lado][:, 2], baja - c["root_l"][:, 2])
    secundario(c, loop, fn.__name__)
    return c


def escribir(nombre, c, n):
    act = bpy.data.actions.new(nombre)
    act.use_fake_user = True
    act.use_frame_range = True
    act.frame_start, act.frame_end = 0, n
    slot = act.slots.new(id_type="OBJECT", name=arm.name)
    bolsa = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    frames = n + 1
    poses = [resolver(c, i) for i in range(frames)]

    def curva(ruta, idx, valores, grupo):
        fc = bolsa.fcurves.new(ruta, index=idx, group_name=grupo)
        fc.keyframe_points.add(frames)
        co = np.empty(frames * 2)
        co[0::2] = np.arange(frames)
        co[1::2] = valores
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * frames)   # LINEAR
        fc.update()

    for b in ROT:
        ruta = 'pose.bones["%s"].' % b
        qs, prev = [], None
        for i in range(frames):
            q = poses[i][b][0].copy()
            if prev is not None and q.dot(prev) < 0:
                q.negate()
            qs.append(list(q)); prev = q
        qs = np.array(qs)
        for j in range(4):
            curva(ruta + "rotation_quaternion", j, qs[:, j], b)
        ls = np.array([list(poses[i][b][1]) for i in range(frames)])
        ss = np.array([list(poses[i][b][2]) for i in range(frames)])
        for j in range(3):
            curva(ruta + "location", j, ls[:, j], b)
            curva(ruta + "scale", j, ss[:, j], b)
    return act


# Se puede volver a pasar sobre el propio Gallina_Final_anim.blend: las Actions de antes se sustituyen.
if arm.animation_data:
    arm.animation_data.action = None
for nombre in CLIPS:
    viejo = bpy.data.actions.get(nombre)
    if viejo is not None:
        bpy.data.actions.remove(viejo)

for nombre, (fn, n, loop) in CLIPS.items():
    act = escribir(nombre, canales(fn, n, loop), n)
    act.use_cyclic = loop
    print("CLIP %-12s %3d f  %.2f s  %s" % (nombre, n, n / FPS, "bucle" if loop else ""))

# ---------------------------------------------------------------- salida
ad = arm.animation_data_create()
ad.action = None
for pb in arm.pose.bones:
    pb.matrix_basis.identity()
blend, fbx = out_base + ".blend", out_base + ".fbx"
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.export_scene.fbx(
    filepath=fbx, object_types={"ARMATURE", "MESH"}, use_selection=False,
    apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
    add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
    use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
    bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
    bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE",
)
print("ANIM_OK", len(CLIPS), blend, fbx)
