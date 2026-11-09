"""Crea las animaciones del gavilan sobre el rig, por linea de comandos.

    blender --background --disable-autoexec "<rig>.blend" --python animar_gavilan.py -- "Gavilan_Final_anim"

(<rig>.blend sale de rig_gavilan.py y es temporal: el repo solo guarda Gavilan_Final_anim.blend/.fbx)

Genera <salida>.blend (una Action por clip) y <salida>.fbx (malla + esqueleto + todos los clips).
Todo va en el sitio: el movimiento y la inclinacion por velocidad los pone HawkController.
En vuelo el centro del cuerpo se queda en HOMBRO_Z, asi los clips de vuelo y de suelo enlazan sin saltos.

Convencion de angulos (grados, ejes del rig en reposo; mira a -Y, su izquierda es +X):
    rx + : lo de arriba cae hacia delante (lo que apunta atras sube)
    ry + : lo de arriba se inclina a su izquierda
    rz + : gira hacia su izquierda
Ala izquierda: rz + sube el ala, ry + la barre hacia atras, rx + baja el borde de ataque.
"""
import bpy, sys, math
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector
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


# ---------------------------------------------------------------- curvas
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


def mezcla(a, b, f):
    return a + (b - a) * f


def pulso(t, t0, dur):
    return sin(math.pi * min(1.0, max(0.0, (t - t0) / dur)))


def parpadeo(t, t0, dur=0.13):
    return 1 - 0.93 * pulso(t, t0, dur)


def suma(d, hueso, r):
    a = d.get(hueso, (0, 0, 0))
    d[hueso] = tuple(x + y for x, y in zip(a, r))


# ---------------------------------------------------------------- poses base
PITCH = 62       # inclinacion del cuerpo en vuelo: asi las alas quedan planas y las patas se recogen solas


def cuerpo(d, pitch, mira=8.0):
    """Inclina el cuerpo y reparte la correccion por el cuello. mira = cuanto mira la cabeza hacia abajo."""
    suma(d, "hips", (pitch, 0, 0))
    suma(d, "spine", (3, 0, 0))
    suma(d, "chest", (3, 0, 0))
    c = pitch + 6 - mira
    suma(d, "neck_01", (-c * 0.24, 0, 0))
    suma(d, "neck_02", (-c * 0.33, 0, 0))
    suma(d, "head", (-c * 0.43, 0, 0))


def ala(d, lado, sube=0.0, barre=0.0, tuerce=0.0, codo=0.0, mano=0.0, punta=0.0, pliega=0.0, cierra=None):
    """sube/barre/tuerce mueven el hombro; codo y mano doblan el ala hacia abajo en cada articulacion.
    pliega va de 0 (extendida) a 1 (recogida contra el costado). cierra = 0..1 junta las plumas."""
    k = 1 if lado == "L" else -1
    p = pliega
    suma(d, "wing_01." + lado, (tuerce, barre * k, sube * k))
    suma(d, "wing_02." + lado, (0, 10 * p * k, (-codo - 7 * p) * k))
    suma(d, "wing_03." + lado, (0, 14 * p * k, (-mano - 9 * p) * k))
    suma(d, "wing_04." + lado, (0, 10 * p * k, (-punta - 5 * p) * k))
    d["recoge." + lado] = max(d.get("recoge." + lado, 0.0), p)
    d["pluma." + lado] = d.get("pluma." + lado, 0.0) + (p if cierra is None else cierra)


def alas(d, sube=0.0, barre=0.0, tuerce=0.0, codo=0.0, mano=0.0, punta=0.0, pliega=0.0, cierra=None):
    for lado in ("L", "R"):
        ala(d, lado, sube, barre, tuerce, codo, mano, punta, pliega, cierra)


def aleteo(d, f, amp, media=8.0, recoge=0.5, tuerce=20.0):
    """f = fase 0..1 (0 arriba del todo, 0.5 abajo). En la subida el ala se dobla para no frenar."""
    a = cos(TAU * f)
    sube_ala = max(0.0, -sin(TAU * f))     # 1 a mitad de la subida
    baja_ala = max(0.0, sin(TAU * f))
    alas(d, sube=media + amp * a, barre=6 * sube_ala * recoge * 2 - 4 * baja_ala,
         tuerce=tuerce + 12 * baja_ala - 16 * sube_ala,
         codo=62 * recoge * sube_ala, mano=44 * recoge * sube_ala, punta=20 * recoge * sube_ala,
         cierra=0.4 * recoge * sube_ala)
    return a


def bote(d, f, alto, gana=0.06):
    """El cuerpo reacciona al ala: sube al bajar el ala. Tambien se estira en el golpe."""
    a = sin(TAU * f - 1.2)
    suma(d, "root_l", (0, 0, alto * a))
    d["estira"] = d.get("estira", 1.0) + gana * sin(TAU * f)


def patas_recogidas(d, b=1.0, garra=30.0):
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (22 * b, 0, 0))
        d["foot." + lado] = d.get("foot." + lado, 0.0) + 74 * b
        d["garra." + lado] = d.get("garra." + lado, 0.0) + garra * b


def patas_colgando(d, pitch, abre=0.0, garra=52.0, delante=0.0):
    """Patas verticales respecto al mundo: con presa o a punto de agarrar."""
    for lado, k in (("L", 1), ("R", -1)):
        suma(d, "leg." + lado, (-pitch - delante, 0, 0))
        d["foot." + lado] = d.get("foot." + lado, 0.0) + delante * 0.4
        d["garra." + lado] = d.get("garra." + lado, 0.0) + garra
        suma(d, "leg." + lado, (0, 8 * abre * k, 0))


def en_suelo(d, b=1.0):
    d["ik"] = b


def mirar(d, y, p=0.0, g=None):
    """Giro seco de cabeza: y = hacia su izquierda, p = hacia abajo."""
    suma(d, "neck_01", (0, 0, y * 0.12))
    suma(d, "neck_02", (p * 0.2, 0, y * 0.28))
    suma(d, "head", (p * 0.8, (g if g is not None else y * 0.2), y * 0.6))


# ---------------------------------------------------------------- vuelo
@clip(2.4, loop=True)
def Hover(t, T):
    """Parado en el aire: aleteo amplio, la cabeza quieta buscando presas."""
    d = {}
    f = 3 * t / T
    cuerpo(d, 34, mira=12)
    aleteo(d, f, 46, media=14, recoge=0.55, tuerce=-4)
    bote(d, f, 0.026)
    y = K(t, (0, 0), (0.42, 0), (0.5, 30, "o"), (1.1, 30), (1.18, -24, "o"), (1.86, -24), (1.94, 0, "o"), (2.4, 0))
    mirar(d, y, 10 * pulso(t, 0.5, 0.7))
    suma(d, "tail_01", (-30, 0, 0))
    d["fan"] = 16
    patas_colgando(d, 34, abre=0.3, garra=8, delante=-22)
    d["eye"] = min(parpadeo(t, 1.0), parpadeo(t, 2.0))
    return d


@clip(0.6, loop=True)
def Fly(t, T):
    d = {}
    f = t / T
    cuerpo(d, PITCH, mira=8)
    a = aleteo(d, f, 40, media=9, recoge=0.5)
    bote(d, f, 0.022)
    suma(d, "hips", (2.5 * a, 0, 0))
    suma(d, "head", (-2.5 * a, 0, 0))
    suma(d, "tail_01", (-27 + 5 * a, 0, 0))
    d["fan"] = 5
    patas_recogidas(d)
    return d


@clip(3.0, loop=True)
def Glide(t, T):
    """Planeo: alas abiertas y la cabeza vigilando abajo."""
    d = {}
    p = t / T
    cuerpo(d, PITCH + 2, mira=10)
    b = sin(TAU * p)
    alabeo = 7 * sin(TAU * p + 0.6)
    suma(d, "hips", (1.5 * sin(2 * TAU * p), 0, 0))
    suma(d, "root_l", (0, 0, 0.012 * sin(2 * TAU * p)))
    for lado, k in (("L", 1), ("R", -1)):
        ala(d, lado, sube=11 + 3 * sin(2 * TAU * p) + alabeo * k * 0.5, tuerce=17 + 3 * b * k,
            codo=-4 + 3 * sin(2 * TAU * p + 1), mano=-6 + 4 * sin(2 * TAU * p + 0.4), punta=-5 + 5 * sin(2 * TAU * p + 0.1))
    y = K(t, (0, 0), (0.6, 0), (0.68, -26, "o"), (1.4, -26), (1.48, 22, "o"), (2.3, 22), (2.38, 0, "o"), (3.0, 0))
    mirar(d, y, 12)
    suma(d, "tail_01", (-27 + 3 * sin(2 * TAU * p), 0, 6 * sin(TAU * p + 2)))
    d["fan"] = 8 + 4 * sin(2 * TAU * p)
    patas_recogidas(d)
    d["eye"] = min(parpadeo(t, 0.62), parpadeo(t, 2.32))
    return d


@clip(0.4, loop=True)
def Climb(t, T):
    """Ganando altura: aleteo corto y furioso con el cuerpo levantado."""
    d = {}
    f = t / T
    cuerpo(d, 42, mira=0)
    a = aleteo(d, f, 52, media=16, recoge=0.65, tuerce=8)
    bote(d, f, 0.03, gana=0.09)
    suma(d, "hips", (4 * a, 0, 0))
    suma(d, "tail_01", (-30 + 8 * a, 0, 0))
    d["fan"] = 20
    patas_recogidas(d, 0.8, 22)
    d["eye"] = 1.06
    d["ceja"] = 0.5
    return d


@clip(0.3, loop=True)
def Boost(t, T):
    """Aceleron: alas medio barridas a toda velocidad, cuerpo estirado como una flecha."""
    d = {}
    f = t / T
    cuerpo(d, PITCH + 16, mira=2)
    a = aleteo(d, f, 34, media=6, recoge=0.35, tuerce=14)
    alas(d, barre=30)
    bote(d, f, 0.012, gana=0.03)
    d["estira"] = d.get("estira", 1.0) + 0.1
    suma(d, "tail_01", (-29 + 3 * a, 0, 0))
    d["fan"] = -6
    patas_recogidas(d, 1, 40)
    d["eye"] = 0.8
    d["ceja"] = 1.0
    d["crest"] = -18
    return d


# ---------------------------------------------------------------- picado
def pose_picado(d, x=1.0):
    # En picado la cabeza sigue al cuerpo y mira a la presa: doblar el cuello 90 grados estiraba la garganta.
    cuerpo(d, mezcla(PITCH, 80, x), mira=mezcla(8, 45, x))
    alas(d, tuerce=mezcla(20, 0, x), pliega=x)
    d["aero"] = 1.0
    suma(d, "tail_01", (mezcla(-27, -39, x), 0, 0))
    d["estira"] = d.get("estira", 1.0) + 0.16 * x
    d["fan"] = d.get("fan", 0.0) - 8 * x
    patas_recogidas(d, 1, mezcla(30, 44, x))
    d["ceja"] = d.get("ceja", 0.0) + x
    d["crest"] = d.get("crest", 0.0) - 26 * x


@clip(0.33)
def Dive_In(t, T):
    """Se encoge hacia arriba, abre las alas y se lanza."""
    d = {}
    b = K(t, (0, 0), (0.13, -0.42), (0.26, 1, "o"), (0.33, 1))
    x = max(0.0, b)
    a = max(0.0, -b) / 0.42
    pose_picado(d, x)
    suma(d, "hips", (-20 * a, 0, 0))
    alas(d, sube=38 * a, barre=-14 * a, tuerce=-14 * a)
    d["estira"] = d.get("estira", 1.0) - 0.12 * a
    d["eye"] = K(t, (0, 1), (0.13, 1.28), (0.26, 0.74, "o"), (0.33, 0.78))
    return d


@clip(0.4, loop=True)
def Dive(t, T):
    """Misil: plumas temblando y la mirada clavada en la presa."""
    d = {}
    p = t / T
    pose_picado(d, 1)
    v = sin(3 * TAU * p)
    suma(d, "hips", (0.9 * sin(5 * TAU * p), 1.6 * v, 0))
    suma(d, "head", (-0.9 * sin(5 * TAU * p), -1.6 * v, 0))
    suma(d, "tail_01", (2.2 * sin(4 * TAU * p + 1), 0, 2.4 * sin(5 * TAU * p)))
    suma(d, "tail_02", (3 * sin(6 * TAU * p + 2), 0, 0))
    for lado, k in (("L", 1), ("R", -1)):
        suma(d, "wing_04." + lado, (0, 5 * sin(6 * TAU * p + k) * k, 0))
        suma(d, "wing_03." + lado, (0, 2.5 * sin(4 * TAU * p + 1 + k) * k, 0))
    d["crest"] = d.get("crest", 0.0) + 4 * sin(7 * TAU * p)
    d["eye"] = 0.78
    return d


@clip(0.6)
def Dive_Out(t, T):
    """Frenazo: las alas se abren de golpe, la cola se despliega y remonta."""
    d = {}
    x = K(t, (0, 1), (0.1, 0, "o"))
    b = K(t, (0, 0), (0.12, 1, "o"), (0.34, 1), (0.6, 0))
    d["aero"] = 1.0
    pitch = K(t, (0, 80), (0.14, 14, "o"), (0.34, 22), (0.6, PITCH))
    cuerpo(d, pitch, mira=K(t, (0, 0), (0.14, -8), (0.6, 8)))
    f = K(t, (0.08, 0), (0.6, 1.0))
    alas(d, pliega=x, tuerce=mezcla(-30, 20, f) * (1 - x))
    if t > 0.08:
        aleteo(d, f, mezcla(0, 44, min(1.0, f * 4)), media=14 * (1 - f) + 8, recoge=0.4, tuerce=0)
    else:
        alas(d, sube=8 * (1 - x))
    suma(d, "tail_01", (mezcla(-39, -8, b) * (1 - f) + -27 * f, 0, 0))
    d["fan"] = 34 * b
    d["estira"] = 1 + K(t, (0, 0.16), (0.1, -0.16), (0.24, 0.05), (0.6, 0))
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (mezcla(22, -70, b), 0, 0))
        d["foot." + lado] = mezcla(74, 0, b)
        d["garra." + lado] = mezcla(30, -30, b)
    d["ceja"] = K(t, (0, 1), (0.14, -0.6), (0.6, 0))
    d["eye"] = K(t, (0, 0.78), (0.12, 1.35, "o"), (0.45, 1.1), (0.6, 1))
    d["crest"] = K(t, (0, -26), (0.14, 30, "o"), (0.6, 0))
    return d


# ---------------------------------------------------------------- caza
PITCH_CARGA = 40


def pose_carga(d, t, T):
    """Comun a Carry y a su forcejeo: aleteo pesado y patas abajo con la presa."""
    f = t / T
    cuerpo(d, PITCH_CARGA, mira=6)
    a = aleteo(d, f, 50, media=15, recoge=0.45, tuerce=2)
    bote(d, f, 0.036, gana=0.07)
    suma(d, "hips", (5 * a, 0, 0))
    suma(d, "tail_01", (-27 + 7 * a, 0, 0))
    d["fan"] = 26
    patas_colgando(d, PITCH_CARGA + 5 * a, garra=56)
    return a


@clip(0.5)
def Grab(t, T):
    """Las garras se adelantan abiertas, se cierran de golpe y el peso de la presa tira de el."""
    d = {}
    g = K(t, (0, 0), (0.1, 1, "o"), (0.2, 1), (0.42, 0))
    c = K(t, (0, 0), (0.18, 0), (0.24, 1, "o"), (0.5, 1))
    pitch = K(t, (0, PITCH), (0.1, 12, "o"), (0.22, 8), (0.42, PITCH_CARGA), (0.5, PITCH_CARGA))
    cuerpo(d, pitch, mira=K(t, (0, 8), (0.12, 26), (0.24, 34), (0.5, 6)))
    alas(d, sube=K(t, (0, 12), (0.12, 58, "o"), (0.24, 48), (0.34, -18), (0.5, 65)),
         tuerce=K(t, (0, 20), (0.12, -26), (0.34, 8), (0.5, 2)),
         mano=K(t, (0, 0), (0.24, -8), (0.34, 32), (0.5, 0)), punta=K(t, (0, 0), (0.34, 26), (0.5, 0)))
    suma(d, "tail_01", (K(t, (0, -27), (0.12, -6), (0.5, -27)), 0, 0))
    d["fan"] = K(t, (0, 5), (0.12, 40, "o"), (0.5, 26))
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (mezcla(22, -pitch - 40 * (1 - c), max(g, c)), 0, 0))
        d["foot." + lado] = mezcla(74, 12 * (1 - c), max(g, c))
        d["garra." + lado] = mezcla(mezcla(30, -34, g), 62, c)
    d["root_l"] = (0, 0, K(t, (0, 0), (0.1, 0.03), (0.26, -0.075, "o"), (0.5, 0.0)))
    d["estira"] = K(t, (0, 1), (0.1, 1.16), (0.24, 0.8), (0.36, 1.1), (0.5, 1.0))
    d["eye"] = K(t, (0, 1), (0.1, 1.3, "o"), (0.24, 0.72), (0.5, 1))
    d["ceja"] = K(t, (0, 0.2), (0.1, 1.0), (0.5, 0.3))
    d["jaw"] = 22 * pulso(t, 0.16, 0.3)
    return d


@clip(0.6, loop=True)
def Carry(t, T):
    """Con presa: cada aletazo cuesta y las garras no se abren."""
    d = {}
    a = pose_carga(d, t, T)
    d["ceja"] = 0.45
    d["jaw"] = 6 + 5 * a
    mirar(d, 0, 10 + 4 * a)
    return d


@clip(0.4)
def Carry_Struggle(t, T):
    """Forcejeo: tirones hacia los lados y un picotazo de aviso a la presa."""
    d = {}
    # Dos aletazos enteros: empieza y acaba exactamente en la pose de Carry.
    pose_carga(d, (2 * t / T % 1) * 0.6, 0.6)
    c = pulso(t, 0.0, 0.4)
    z = sin(TAU * 2 * t / T) * c
    suma(d, "hips", (0, 10 * z, 7 * z))
    suma(d, "chest", (0, -7 * z, -4 * z))
    pico = K(t, (0, 0), (0.08, -0.45), (0.17, 1, "o"), (0.28, 1), (0.4, 0))
    suma(d, "neck_01", (15 * pico, 0, 0))
    suma(d, "neck_02", (21 * pico, 0, 0))
    suma(d, "head", (36 * pico, 7 * z, 0))
    d["jaw"] = 6 + 24 * max(0.0, -pico) / 0.45 + 5 * c
    d["ceja"] = 0.45 + 0.55 * c
    d["eye"] = 1 - 0.18 * c
    for lado in ("L", "R"):
        d["garra." + lado] = d.get("garra." + lado, 0.0) + 14 * c
    return d


@clip(0.5)
def Drop(t, T):
    """Abre las garras, la presa cae y el cuerpo sale despedido hacia arriba."""
    d = {}
    f = K(t, (0, 0), (0.5, 1))
    o = K(t, (0, 0), (0.08, 1, "o"), (0.32, 1), (0.5, 0))
    pitch = mezcla(PITCH_CARGA, PITCH, ease(t / T))
    cuerpo(d, pitch, mira=K(t, (0, 6), (0.16, 30), (0.5, 8)))
    aleteo(d, ((0.5 * t / T) + 0.5 * (t / T) ** 2) % 1.0, 40 + 8 * o, media=12, recoge=0.45, tuerce=mezcla(2, 20, f))
    suma(d, "tail_01", (-27, 0, 0))
    d["fan"] = mezcla(26, 5, f)
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (mezcla(-pitch, 22, ease((t - 0.12) / 0.38)), 0, 0))
        d["foot." + lado] = mezcla(0, 74, ease((t - 0.12) / 0.38))
        d["garra." + lado] = mezcla(mezcla(56, -38, min(1.0, t / 0.06)), 30, ease((t - 0.22) / 0.28))
    d["root_l"] = (0, 0, 0.1 * pulso(t, 0.0, 0.5))
    d["estira"] = K(t, (0, 1), (0.1, 1.12), (0.28, 0.96), (0.5, 1))
    d["eye"] = K(t, (0, 1), (0.1, 1.12), (0.5, 1))
    return d


@clip(1.1)
def Screech(t, T):
    """El gran gesto: carga aire, adelanta la cabeza y lo suelta con todo el cuerpo."""
    d = {}
    # 0..1 carga, 1..2 chillido sostenido, 2 vuelta
    c = K(t, (0, 0), (0.2, 1), (0.28, 0), (1.1, 0))
    g = K(t, (0.2, 0), (0.28, 1, "o"), (0.78, 1), (1.1, 0))
    v = g * sin(TAU * 13 * t)
    cuerpo(d, 34 + 12 * g - 14 * c, mira=12 - 26 * c + 4 * g)
    f = 3 * t / T
    aleteo(d, f, 46 * (1 - g) + 9 * g, media=14 + 30 * g - 8 * c, recoge=0.55 * (1 - g), tuerce=-4)
    if g > 0:
        alas(d, barre=-22 * g, tuerce=-18 * g)
    bote(d, f, 0.026 * (1 - g))
    # El cuello carga hacia atras y dispara hacia delante.
    suma(d, "neck_01", (-14 * c + 14 * g, 0, 0))
    suma(d, "neck_02", (-14 * c + 8 * g + 1.5 * v, 0, 0))
    suma(d, "head", (-10 * c - 26 * g, 2.5 * v, 0))
    suma(d, "chest", (-7 * c + 3 * g, 0, 0))
    d["root_l"] = (0, K(t, (0, 0), (0.2, 0.03), (0.28, -0.05, "o"), (1.1, 0)), d.get("root_l", (0, 0, 0))[2])
    d["jaw"] = 8 * c + 58 * g + 3 * v
    d["jaw_s"] = 1 + 0.3 * g
    d["crest"] = 44 * g + 6 * v - 14 * c
    d["ceja"] = -0.3 * c + 1.0 * g
    d["eye"] = 1 + 0.25 * c - 0.1 * g
    d["fan"] = 16 + 36 * g
    suma(d, "tail_01", (-30 + 20 * g, 0, 0))
    d["estira"] = d.get("estira", 1.0) - 0.1 * c + 0.1 * g
    patas_colgando(d, 34 + 12 * g, abre=0.3 + 0.8 * g, garra=8 - 40 * g, delante=-22 + 22 * g)
    return d


# ---------------------------------------------------------------- derribo
@clip(0.4)
def Shot(t, T):
    """El perdigonazo: latigazo con los ojos como platos y todo el cuerpo descolocado."""
    d = {}
    b = K(t, (0, 0), (0.05, 1, "o"), (0.4, 0.75))
    cuerpo(d, mezcla(PITCH, -8, b), mira=8)
    alas(d, sube=mezcla(10, 68, b), barre=-28 * b, tuerce=mezcla(20, -34, b), codo=-30 * b, mano=38 * b, punta=30 * b)
    suma(d, "hips", (0, 0, 24 * b))
    suma(d, "neck_02", (24 * b, 0, -20 * b))
    suma(d, "head", (-36 * b, 14 * b, -18 * b))
    suma(d, "tail_01", (mezcla(-27, 16, b), 0, 0))
    d["fan"] = 44 * b
    d["eye.L"] = K(t, (0, 1), (0.05, 1.6, "o"), (0.4, 1.42))
    d["eye.R"] = K(t, (0, 1), (0.05, 1.32, "o"), (0.4, 1.12))
    d["jaw"] = 42 * b
    d["ceja"] = -1.0 * b
    d["crest"] = 52 * b
    d["estira"] = K(t, (0, 1), (0.05, 1.3), (0.15, 0.82), (0.26, 1.08), (0.4, 1))
    for lado, k in (("L", 1), ("R", -1)):
        suma(d, "leg." + lado, (mezcla(22, -40 * k, b), 0, 0))
        d["foot." + lado] = 74 * (1 - b)
        d["garra." + lado] = -36 * b
    return d


@clip(0.6, loop=True)
def Fall(t, T):
    """Cae dando vueltas como un saco de plumas."""
    d = {}
    p = t / T
    d["hips"] = (360 * p, 0, 18 * sin(TAU * p))
    a = sin(2 * TAU * p)
    alas(d, sube=28 + 30 * a, barre=-12, tuerce=-30, codo=-24 * a, mano=30 + 26 * a, punta=26 * a)
    d["jaw"] = 30 + 8 * a
    suma(d, "neck_02", (10 * sin(TAU * p), 0, 16 * sin(TAU * p + 1)))
    suma(d, "head", (-14, 14 * sin(TAU * p), 18 * sin(TAU * p + 2)))
    suma(d, "tail_01", (12 * a, 0, 0))
    d["fan"] = 30
    d["eye.L"], d["eye.R"] = 1.34, 0.42
    d["ceja"] = -1.0
    d["crest"] = 40
    for lado, k in (("L", 1), ("R", -1)):
        suma(d, "leg." + lado, (-44 + 34 * sin(2 * TAU * p + k), 0, 12 * k))
        d["foot." + lado] = 22 * sin(2 * TAU * p + 1 + k)
        d["garra." + lado] = -30
    return d


ALTO_ESPALDA = -0.11    # hips_l z con el gavilan tumbado boca arriba


def boca_arriba(d, b=1.0):
    """Tumbado de espaldas: pico al cielo, alas abiertas en el suelo, cola estirada y patas al aire."""
    suma(d, "hips", (-90 * b, 0, 0))
    suma(d, "hips_l", (0, 0, ALTO_ESPALDA * b))
    alas(d, sube=-12 * b, barre=-6 * b, codo=-8 * b, mano=-6 * b)
    suma(d, "tail_01", (-54 * b, 0, 0))
    suma(d, "tail_02", (-4 * b, 0, 0))
    suma(d, "neck_01", (5 * b, 0, 0))
    suma(d, "neck_02", (5 * b, 0, 0))
    suma(d, "head", (6 * b, 24 * b, 0))
    for lado, k in (("L", 1), ("R", -1)):
        suma(d, "leg." + lado, (-62 * b, 0, 6 * k * b))
        d["foot." + lado] = d.get("foot." + lado, 0.0) - 18 * b
        d["garra." + lado] = d.get("garra." + lado, 0.0) + 34 * b


@clip(0.5)
def Crash(t, T):
    """Se estampa, rebota una vez y se queda patas arriba."""
    d = {}
    b = K(t, (0, 0), (0.08, 1))
    boca_arriba(d, b)
    bot = 0.13 * pulso(t, 0.1, 0.24)
    # Mientras gira, la cola barreria el suelo: el golpe lo levanta.
    suma(d, "root_l", (0, 0, bot + 0.05 * pulso(t, 0.32, 0.14) + 0.2 * K(t, (0, 1), (0.1, 0))))
    d["scale"] = K(t, (0, 1), (0.08, 0.62), (0.2, 1.18), (0.33, 0.8), (0.42, 1.06), (0.5, 1))
    k = K(t, (0, 0), (0.08, 1), (0.2, -0.5), (0.33, 0.6), (0.5, 0))
    alas(d, sube=-34 * k + 20 * pulso(t, 0.1, 0.24), mano=40 * pulso(t, 0.1, 0.24))
    suma(d, "head", (20 * k, 0, 0))
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (-24 * k, 0, 0))
        d["garra." + lado] = d.get("garra." + lado, 0.0) - 30 * pulso(t, 0.08, 0.3)
    d["eye.L"] = K(t, (0, 1.34), (0.08, 0.2), (0.34, 1.3), (0.5, 1.1))
    d["eye.R"] = K(t, (0, 0.42), (0.08, 0.2), (0.34, 0.5), (0.5, 0.55))
    d["jaw"] = K(t, (0, 30), (0.08, 44), (0.5, 24))
    d["ceja"] = -0.8
    d["crest"] = K(t, (0, 40), (0.08, -20), (0.2, 50), (0.5, 22))
    d["fan"] = 26
    return d


@clip(2.0, loop=True)
def Downed(t, T):
    """KO: patas al aire con un tembleque y la cabeza dando vueltas."""
    d = {}
    a = TAU * t / T
    boca_arriba(d)
    suma(d, "neck_02", (0, 7 * sin(a), 7 * cos(a)))
    suma(d, "head", (8 * sin(2 * a), 14 * sin(a - 0.5), 16 * cos(a - 0.5)))
    d["jaw"] = 24 + 4 * sin(2 * a)
    d["eye.L"] = 1.1 + 0.2 * sin(a)
    d["eye.R"] = 0.55 - 0.2 * sin(a)
    d["ceja"] = -0.8
    d["crest"] = 22 + 6 * sin(2 * a)
    d["fan"] = 26
    # Una pata tiembla a ratos.
    tic = pulso(t, 0.5, 0.42)
    tic2 = pulso(t, 1.4, 0.34)
    v = sin(TAU * 11 * t)
    suma(d, "leg.L", (-6 * tic * v, 0, 0))
    d["garra.L"] = d.get("garra.L", 0.0) - 50 * tic * (0.5 + 0.5 * v)
    suma(d, "leg.R", (-6 * tic2 * v, 0, 0))
    d["garra.R"] = d.get("garra.R", 0.0) - 50 * tic2 * (0.5 + 0.5 * v)
    alas(d, mano=7 * tic2 * v, punta=9 * tic * v)
    return d


@clip(0.95)
def Recover(t, T):
    """Se endereza de un salto, se sacude las plumas y vuelve al aire."""
    d = {}
    # 0-0.28 se pone de pie, 0.28-0.6 se sacude, 0.6-0.95 despega
    b = K(t, (0, 1), (0.1, 1.08), (0.28, 0, "o"))
    sac = pulso(t, 0.26, 0.36)
    v = sac * sin(TAU * 8.5 * t)
    vuelo = K(t, (0.58, 0), (0.86, 1))
    boca_arriba(d, b)
    suelo = (1 - b) * (1 - vuelo)
    cuerpo(d, PITCH * vuelo, mira=8 * vuelo)
    plegada = K(t, (0, 0), (0.26, 0.82), (0.6, 0.82), (0.72, 0))
    if vuelo > 0.01:
        aleteo(d, K(t, (0.58, 0.1), (0.95, 1.75)), 44 * vuelo, media=10, recoge=0.5, tuerce=20 * vuelo)
    alas(d, pliega=plegada, sube=22 * sac + 8 * v, mano=8 * v)
    suma(d, "hips", (0, 7 * v, 11 * v))
    suma(d, "chest", (0, -6 * v, -9 * v))
    suma(d, "head", (0, 9 * v, 16 * v))
    suma(d, "tail_01", (6 * sac * (1 - vuelo) + -27 * vuelo, 0, -18 * v))
    d["fan"] = 26 * (1 - min(1, t / 0.3)) + 34 * sac + 5 * vuelo
    d["crest"] = K(t, (0, 22), (0.3, 30), (0.6, 40), (0.72, 0))
    d["eye.L"] = K(t, (0, 1.1), (0.3, 1.2), (0.62, 1))
    d["eye.R"] = K(t, (0, 0.55), (0.3, 1.2), (0.62, 1))
    d["jaw"] = K(t, (0, 24), (0.26, 6), (0.62, 0))
    d["ceja"] = K(t, (0, -0.8), (0.62, 0.2), (0.95, 0))
    d["estira"] = 1 + 0.04 * sac * cos(TAU * 8.5 * t)
    d["root_l"] = (0, 0, d.get("root_l", (0, 0, 0))[2] + 0.2 * pulso(t, -0.04, 0.34) + 0.1 * pulso(t, 0.56, 0.5))
    d["ik"] = suelo
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (22 * vuelo, 0, 0))
        d["foot." + lado] = d.get("foot." + lado, 0.0) + 74 * vuelo
        d["garra." + lado] = d.get("garra." + lado, 0.0) + 30 * vuelo
    return d


# ---------------------------------------------------------------- suelo
def de_pie(d, plegada=1.0):
    alas(d, pliega=plegada)
    en_suelo(d)


@clip(3.0, loop=True)
def Idle(t, T):
    """De pie y orgulloso (seleccion de personaje, lobby y victoria en el nido)."""
    d = {}
    b = sin(TAU * 2 * t / T)
    de_pie(d)
    d["scale"] = 1 + 0.012 * b
    suma(d, "hips_l", (0, 0, 0.003 * b))
    suma(d, "chest", (1.4 * b, 0, 0))
    y = K(t, (0, 0), (0.5, 0), (0.58, 34, "o"), (1.3, 34), (1.38, -22, "o"), (2.1, -22), (2.18, 8, "o"), (2.78, 8), (3.0, 0))
    p = K(t, (0, 0), (1.3, 0), (1.38, 12, "o"), (2.1, 12), (2.18, -4, "o"), (2.78, -4), (3.0, 0))
    mirar(d, y, p)
    # Sacudida de cola y estiron de ala.
    w = pulso(t, 1.75, 0.5)
    suma(d, "tail_01", (5 * w, 0, 13 * sin(TAU * 7 * t) * w))
    d["fan"] = 12 * w
    ala(d, "L", sube=-4 * w, tuerce=3 * w)
    ala(d, "R", sube=-4 * w, tuerce=3 * w)
    d["eye"] = min(parpadeo(t, 0.95), parpadeo(t, 2.5))
    d["ceja"] = 0.25
    return d


@clip(0.6)
def Takeoff(t, T):
    """Se agacha, salta y el primer aletazo lo lleva a la pose de subida."""
    d = {}
    agacha = K(t, (0, 0), (0.14, 1), (0.22, 0))
    sale = K(t, (0.14, 0), (0.34, 1))
    pitch = K(t, (0.14, 0), (0.42, 42))
    cuerpo(d, pitch + 18 * agacha, mira=8 * agacha)
    plegada = K(t, (0, 1), (0.14, 0.45), (0.22, 0))
    f = K(t, (0.14, 0.0), (0.6, 1.5))
    alas(d, pliega=plegada)
    aleteo(d, f, 52 * sale, media=16 * sale + 26 * agacha, recoge=0.65, tuerce=8 * sale)
    suma(d, "hips_l", (0, 0, -0.08 * agacha))
    d["root_l"] = (0, 0, 0.09 * sale)
    d["ik"] = 1 - sale
    d["estira"] = 1 - 0.14 * agacha + 0.16 * pulso(t, 0.14, 0.26)
    suma(d, "tail_01", (-30 * sale + 12 * agacha, 0, 0))
    d["fan"] = 20 * sale + 24 * agacha
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (mezcla(-pitch, 18, ease((t - 0.26) / 0.34)), 0, 0))
        d["foot." + lado] = 60 * ease((t - 0.26) / 0.34)
        d["garra." + lado] = 22 * ease((t - 0.26) / 0.34)
    d["ceja"] = 0.6
    return d


@clip(0.6)
def Land(t, T):
    """Frena con las alas, clava las patas y amortigua hasta quedar de pie."""
    d = {}
    frena = K(t, (0, 0), (0.16, 1, "o"), (0.28, 1), (0.42, 0))
    llega = K(t, (0.16, 0), (0.3, 1))
    pitch = K(t, (0, PITCH), (0.16, 2, "o"), (0.3, 8), (0.44, 0))
    aplasta = K(t, (0.24, 0), (0.3, 1, "o"), (0.42, -0.3), (0.6, 0))
    cuerpo(d, pitch + 16 * max(0.0, aplasta), mira=K(t, (0, 8), (0.3, 18), (0.6, 0)))
    plegada = K(t, (0.34, 0), (0.56, 1, "o"))
    alas(d, pliega=plegada)
    f = K(t, (0, 0.05), (0.34, 1.1))
    aleteo(d, f, 34 * (1 - plegada), media=(10 + 34 * frena) * (1 - plegada), recoge=0.5 * (1 - plegada),
           tuerce=mezcla(20, -34, frena) * (1 - plegada))
    suma(d, "hips_l", (0, 0, -0.07 * max(0.0, aplasta) + 0.03 * min(0.0, aplasta)))
    d["root_l"] = (0, 0, 0.1 * (1 - llega))
    d["ik"] = llega
    d["estira"] = 1 + 0.05 * frena - 0.1 * max(0.0, aplasta)
    suma(d, "tail_01", (mezcla(-27, 10, frena) * (1 - plegada), 0, 0))
    d["fan"] = 40 * frena
    for lado in ("L", "R"):
        suma(d, "leg." + lado, (mezcla(22, -pitch - 22, ease(t / 0.16)), 0, 0))
        d["foot." + lado] = mezcla(74, -20, ease(t / 0.16)) * (1 - llega)
        d["garra." + lado] = mezcla(30, -28, ease(t / 0.16)) * (1 - llega)
    d["eye"] = K(t, (0, 1), (0.28, 1), (0.34, 0.5), (0.5, 1.05), (0.6, 1))
    return d


@clip(0.9, loop=True)
def Victory(t, T):
    """Saca pecho: alas abiertas, salta y grita al cielo."""
    d = {}
    p = (t / T * 2) % 1.0
    sube = 4 * p * (1 - p)
    giro = sin(TAU * t / T)
    en_suelo(d)
    d["root_l"] = (0, 0, 0.11 * sube)
    d["scale"] = 1 + 0.12 * (sube - 0.55) * (1 if sube > 0.55 else 1.6)
    suma(d, "hips", (-7 * sube, 5 * giro, 13 * giro))
    suma(d, "chest", (-4, -3 * giro, -5 * giro))
    suma(d, "neck_02", (-8 * sube, 0, 0))
    suma(d, "head", (-30 * sube, 10 * giro, 8 * giro))
    d["jaw"] = 8 + 30 * sube
    a = sin(TAU * 3 * t / T * 2)
    alas(d, sube=34 + 22 * a * 0.5 + 12 * sube, tuerce=-10, mano=14 * a, punta=18 * a)
    suma(d, "tail_01", (10 * sube, 0, -11 * giro))
    d["fan"] = 26 + 22 * sube
    d["crest"] = 14 + 22 * sube
    d["eye"] = 0.72
    d["ceja"] = -0.3
    for lado in ("L", "R"):
        d["ankle." + lado] = (0, 0, 0.03 * sube)
        d["garra." + lado] = -16 * sube
    return d


# ---------------------------------------------------------------- gestos (rueda de gestos del juego; en el aire)
@clip(1.6, loop=True)
def Dance(t, T):
    """Baile en el aire: aleteo a tiempo, el cuerpo se balancea de lado a lado, la cabeza cabecea al compas,
    la cresta y la cola se abren en cada golpe."""
    d = {}
    u = t / T
    f = 4 * u                                    # cuatro aleteos por ciclo, uno por tiempo
    beat = (u * 4) % 1.0
    vaiven = sin(TAU * u)
    cuerpo(d, 30, mira=10)
    aleteo(d, f, 50, media=16, recoge=0.5, tuerce=-4)
    bote(d, f, 0.03)
    suma(d, "hips", (0, 14 * vaiven, 10 * vaiven))
    suma(d, "chest", (0, -6 * vaiven, -6 * vaiven))
    cab = K(beat, (0, 0), (0.15, 1, "o"), (0.55, -0.5), (1, 0))
    mirar(d, -18 * vaiven, 14 * cab, g=12 * vaiven)
    suma(d, "tail_01", (-26, 0, -14 * vaiven))
    d["fan"] = 18 + 16 * pulso(beat, 0.0, 0.4)
    d["crest"] = 8 + 22 * pulso(beat, 0.05, 0.35)
    d["jaw"] = 10 * pulso(beat, 0.1, 0.3)
    patas_colgando(d, 30, abre=0.4 + 0.3 * vaiven, garra=-10, delante=-18)
    d["eye"] = 0.85
    return d


@clip(1.4, loop=True)
def Wave(t, T):
    """¡Hola!: reverencia en el aire. Baja el pecho y la cabeza dos veces (como quien saluda con la cabeza), la cresta
    se levanta y abre la cola; el aleteo sigue tranquilo."""
    d = {}
    u = t / T
    f = 3 * u
    inclina = K(u, (0, 0), (0.18, 1, "o"), (0.38, 0.1), (0.55, 1, "o"), (0.78, 0.1), (1, 0))
    cuerpo(d, 34 + 14 * inclina, mira=12)
    aleteo(d, f, 44, media=14, recoge=0.55, tuerce=-4)
    bote(d, f, 0.022)
    mirar(d, 0, 26 * inclina, g=8)
    suma(d, "tail_01", (-30 + 12 * inclina, 0, 0))
    d["fan"] = 20 + 10 * inclina
    d["crest"] = 18 + 14 * inclina
    patas_colgando(d, 34, abre=0.3, garra=8, delante=-22)
    d["eye"] = 0.75
    return d


@clip(1.2, loop=True)
def Taunt(t, T):
    """Burla: se ríe a carcajadas en el aire. Pico que castañetea, cabeza que niega, la cresta de punta y una ceja
    arriba, el cuerpo se sacude con cada carcajada."""
    d = {}
    u = t / T
    f = 3 * u
    carcajada = abs(sin(TAU * u * 5))
    cuerpo(d, 26 - 6 * carcajada, mira=0)
    aleteo(d, f, 42, media=14, recoge=0.55, tuerce=-4)
    bote(d, f, 0.02)
    suma(d, "chest", (-6 * carcajada, 0, 0))
    mirar(d, 22 * sin(TAU * u * 2), -16 - 8 * carcajada)
    d["jaw"] = 8 + 30 * carcajada
    d["crest"] = 30
    d["ceja"] = 0.5
    suma(d, "tail_01", (-24, 0, 16 * sin(TAU * u * 4)))
    d["fan"] = 30
    patas_colgando(d, 26, abre=0.3, garra=6, delante=-20)
    d["eye"] = 0.65
    return d


# ---------------------------------------------------------------- movimiento secundario
GESTOS = {"Dive_In", "Dive_Out", "Grab", "Carry_Struggle", "Drop", "Screech", "Shot", "Crash", "Recover", "Takeoff", "Land"}


def muelle(senal, hz, zeta, bucle, pasadas=4):
    """Devuelve la senal filtrada por un muelle: lo que llega tarde y rebota."""
    n = len(senal)
    w = TAU * hz
    x, v = senal[0], 0.0
    sal = np.zeros(n)
    dt = 1.0 / FPS / 4
    m = n - 1 if bucle else n
    for _ in range(pasadas if bucle else 1):
        for i in range(m):
            for _ in range(4):
                v += (w * w * (senal[i] - x) - 2 * zeta * w * v) * dt
                x += v * dt
            sal[i] = x
    if bucle:
        sal[n - 1] = sal[0]
    return sal


def secundario(c, bucle, nombre):
    """Puntas y bordes de ala, cola y penacho siguen al cuerpo con retraso."""
    n = len(c["hips"])
    g = np.ones(n)
    if nombre in GESTOS:
        k = min(6, n - 1)
        g[-k:] = np.linspace(1, 0, k)
    if nombre == "Fall":
        return      # la vuelta entera no se puede filtrar: el clip ya lleva su propio caos

    alto = c["hips_l"][:, 2] + c["root_l"][:, 2]

    # Alas: la punta llega tarde al aletazo y el borde de fuga se dobla contra el aire.
    for s, k in ((".L", 1), (".R", -1)):
        sube = c["wing_01" + s][:, 2] * k
        lag = muelle(sube, 7.5, 0.32, bucle) - sube
        c["wing_03" + s][:, 2] += k * g * np.clip(lag * 0.45, -40, 40)
        c["wing_04" + s][:, 2] += k * g * np.clip(lag * 0.65, -45, 45)
        vel = np.gradient(sube) * FPS
        flex = muelle(vel, 8.0, 0.4, bucle)
        for f, ga in (("feather_1", 0.022), ("feather_2", 0.03), ("feather_3", 0.04)):
            c[f + s][:, 0] += g * np.clip(-flex * ga, -24, 24)

    # Cola: la punta sigue a la base y a los botes del cuerpo.
    p = c["hips"][:, 0] + c["tail_01"][:, 0] - 260 * alto
    y = c["hips"][:, 2] + c["tail_01"][:, 2]
    lag_p, lag_y = muelle(p, 4.6, 0.2, bucle) - p, muelle(y, 4.6, 0.2, bucle) - y
    c["tail_02"][:, 0] += g * np.clip(lag_p * 0.55, -24, 24)
    c["tail_02"][:, 2] += g * np.clip(lag_y * 0.55, -22, 22)

    # Penacho: se levanta cuando la cabeza frena.
    col = ["hips", "spine", "chest", "neck_01", "neck_02", "head"]
    pitch = np.sum([c[b][:, 0] for b in col], axis=0) + 300 * alto
    c["crest"][:, 0] += g * np.clip((muelle(pitch, 5.2, 0.16, bucle) - pitch) * 0.45, -24, 24)


# ---------------------------------------------------------------- de canales a Actions
scene = bpy.context.scene
scene.render.fps = FPS
arm = bpy.data.objects["Gavilan_Rig"]
bpy.context.view_layer.objects.active = arm
data = arm.data
Rb = {b.name: b.matrix_local.to_quaternion() for b in data.bones}
Hb = {b.name: b.head_local.copy() for b in data.bones}
Tb = {b.name: b.tail_local.copy() for b in data.bones}
ROT = list(data.bones.keys())
DEDOS = ("1", "2", "3", "4")
BAJO = Vector((0, 0, -1))
EJE_DEDO = {}
for s in ("L", "R"):
    for k in DEDOS:
        for tipo in ("toe", "claw"):
            b = "%s_%s.%s" % (tipo, k, s)
            dir_b = (Tb[b] - Hb[b]).normalized()
            dir_b.z = 0
            EJE_DEDO[b] = dir_b.normalized().cross(BAJO).normalized()
_d_cola = (Tb["tail_02"] - Hb["tail_01"]).normalized()
NORMAL_COLA = Vector((0, -_d_cola.z, _d_cola.y))     # perpendicular a la cola, hacia arriba y atras
FAN_ESCALA = 0.35     # 52 grados de clip -> 18: la punta se ensancha ~25 %, no ~80 %


def plegado_izquierda(envergadura):
    """Orientacion del ala recogida: la punta hacia donde se pida, dorso marron hacia fuera y borde de ataque delante."""
    envergadura = Vector(envergadura).normalized()
    dorso = (Vector((1, 0, 0)) - envergadura * envergadura.x).normalized()
    ataque = envergadura.cross(dorso)
    return Matrix((envergadura, dorso, ataque)).transposed().to_quaternion()


ESPEJO = Matrix.Diagonal((-1, 1, 1))
Q_PLIEGUE = {"L": plegado_izquierda((0.22, 0.71, -0.67))}     # posado
Q_PICADO = {"L": plegado_izquierda((0.2, 0.17, -0.965))}       # pegada al eje del cuerpo, que en picado va horizontal
for tabla in (Q_PLIEGUE, Q_PICADO):
    tabla["R"] = (ESPEJO @ tabla["L"].to_matrix() @ ESPEJO).to_quaternion()
SUELA = Vector((0, 0.0, -0.058))


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
    e = max(0.3, float(c["estira"][i]))
    pose["hips"][2] = Vector((1 / math.sqrt(e), e, 1 / math.sqrt(e)))

    ik = min(1.0, max(0.0, float(c["ik"][i])))
    suelas = []
    for lado in ("L", "R"):
        leg, foot = "leg." + lado, "foot." + lado
        q_fk = q_rig(c[leg][i])
        pivote = Hb["hips"]
        cabeza = d_h + pivote + q_h @ (Hb[leg] - pivote)
        tobillo = Hb[foot] + Vector(c["ankle." + lado][i])
        reposo = Hb[foot] - Hb[leg]
        v = tobillo - cabeza
        q_w = reposo.rotation_difference(v)
        delta = tobillo - (cabeza + q_w @ reposo)
        q_ik = q_h.inverted() @ q_w
        q_l = q_fk.slerp(q_ik, ik)
        pose[leg][0] = a_local(leg, q_l)
        pose[leg][1] = Rb[leg].inverted() @ (q_h.inverted() @ delta) * ik
        q_pie = q_rig((c["foot_rx." + lado][i], 0, 0))
        q_pie_ik = (q_h @ q_l).inverted() @ q_pie
        q_f = q_pie.slerp(q_pie_ik, ik)
        pose[foot][0] = a_local(foot, q_f)
        g = float(c["garra." + lado][i])
        for k in DEDOS:
            for tipo, f in (("toe", 0.8), ("claw", 0.75)):
                b = "%s_%s.%s" % (tipo, k, lado)
                # El dedo trasero cierra contra los delanteros con mas fuerza.
                giro = g * f * (1.25 if k == "1" else 1.0)
                pose[b][0] = a_local(b, Quaternion(EJE_DEDO[b], math.radians(giro)))
        suelas.append(Hb[leg] + q_l @ (reposo + q_f @ SUELA))

    # El punto de agarre va siempre entre las dos garras y se mantiene horizontal.
    medio = (suelas[0] + suelas[1]) * 0.5 - Hb["talons"]
    pose["talons"][1] = Rb["talons"].inverted() @ medio
    pose["talons"][0] = a_local("talons", q_h.inverted())

    for lado in ("L", "R"):
        o = float(c["ojo." + lado][i])
        ancho = 1 + (o - 1) * 0.6 if o > 1 else 1.0
        pose["eye." + lado][2] = Vector((ancho, ancho, max(0.06, o)))
        # Ceja: + enfadada (baja hacia el pico), - sorprendida (sube y se abre).
        a = float(c["ceja"][i])
        k = 1 if lado == "L" else -1
        pose["brow." + lado][0] = a_local("brow." + lado, q_rig((0, 12 * a * k, 0)))
        pose["brow." + lado][1] = Rb["brow." + lado].inverted() @ Vector((0, 0, -0.006 * a))
        hombro = "wing_01." + lado
        q_rec = Q_PLIEGUE[lado].slerp(Q_PICADO[lado], float(np.clip(c["aero"][i], 0, 1)))
        rec = float(np.clip(c["recoge." + lado][i], 0, 1))
        pose[hombro][0] = a_local(hombro, q_rig(c[hombro][i]).slerp(q_rec, rec))
        # Recogida, la raiz se mete bajo las plumas del hombro para que no se vea el corte.
        pose[hombro][1] = Rb[hombro].inverted() @ Vector((-0.012 * k, 0, 0.026)) * rec
        p = float(c["pluma." + lado][i])
        for f in ("feather_1.", "feather_2.", "feather_3."):
            pose[f + lado][2] = Vector((1, max(0.35, 1 - 0.55 * p), 1))
    pose["jaw"][2] = Vector((1, float(c["jaw_s"][i]), 1))
    # La cola es una sola pieza: el abanico abre menos que unas plumas sueltas, o los lados se estiran.
    ang = float(c["fan"][i]) * FAN_ESCALA
    # Los lados cuelgan de tail_01 pero el centro dobla con tail_02: girar la mitad de esa flexion desde la base
    # lleva la punta lateral al mismo sitio que la central (asi no se rasga la cola al doblarse).
    q02 = Rb["tail_02"] @ pose["tail_02"][0] @ Rb["tail_02"].inverted()
    mitad = Quaternion().slerp(q02, 0.5)
    for lado, sg in (("L", -1), ("R", 1)):
        b = "tail_fan." + lado
        pose[b][0] = a_local(b, Quaternion(NORMAL_COLA, math.radians(sg * ang)) @ mitad) @ pose[b][0]
    return pose


FLOTANTES = ("foot_rx.L", "foot_rx.R", "garra.L", "garra.R", "fan", "ceja", "pluma.L", "pluma.R", "recoge.L", "recoge.R", "aero", "ik")
UNOS = ("scale", "estira", "ojo.L", "ojo.R", "jaw_s")


def canales(fn, n, loop):
    T = n / FPS
    frames = n + 1
    c = {b: np.zeros((frames, 3)) for b in ROT}
    for k in ("hips_l", "root_l", "ankle.L", "ankle.R"):
        c[k] = np.zeros((frames, 3))
    for k in FLOTANTES:
        c[k] = np.zeros(frames)
    for k in UNOS:
        c[k] = np.ones(frames)
    for i in range(frames):
        t = i / FPS
        d = fn(0.0 if (loop and i == n) else t, T)
        for k, v in d.items():
            if k in ("foot.L", "foot.R"):
                c["foot_rx." + k[-1]][i] = v
            elif k == "jaw":
                c["jaw"][i, 0] = v
            elif k == "crest":
                c["crest"][i, 0] = -v
            elif k == "eye":
                c["ojo.L"][i] = c["ojo.R"][i] = v
            elif k in ("eye.L", "eye.R"):
                c["ojo." + k[-1]][i] = v
            else:
                c[k][i] = v
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


# Cada clip de un solo uso acaba exactamente donde empieza el siguiente, y empieza donde lo dejo el anterior.
SALE_A = {"Dive_In": ("Dive", 3), "Dive_Out": ("Fly", 5), "Grab": ("Carry", 4), "Drop": ("Fly", 4),
          "Carry_Struggle": ("Carry", 4), "Recover": ("Fly", 5), "Takeoff": ("Climb", 5), "Land": ("Idle", 4),
          "Crash": ("Downed", 4), "Shot": ("Fall", 3), "Screech": ("Hover", 4)}
VIENE_DE = {"Recover": ("Downed", 2), "Takeoff": ("Idle", 2), "Land": ("Fly", 2), "Drop": ("Carry", 2),
            "Carry_Struggle": ("Carry", 3), "Dive_Out": ("Dive", 2)}


def fundir(c, ref, indices, pesos):
    for clave, serie in c.items():
        for i, w in zip(indices, pesos):
            serie[i] = serie[i] * (1 - w) + ref[clave] * w


todos = {nombre: canales(fn, n, loop) for nombre, (fn, n, loop) in CLIPS.items()}
for nombre, (destino, k) in SALE_A.items():
    n = CLIPS[nombre][1]
    fundir(todos[nombre], {c: v[0] for c, v in todos[destino].items()}, range(n - k + 1, n + 1), [ease(j / k) for j in range(1, k + 1)])
for nombre, (origen, k) in VIENE_DE.items():
    fundir(todos[nombre], {c: v[0] for c, v in todos[origen].items()}, range(0, k), [1 - ease(j / k) for j in range(k)])

for nombre, (fn, n, loop) in CLIPS.items():
    act = escribir(nombre, todos[nombre], n)
    act.use_cyclic = loop
    print("CLIP %-15s %3d f  %.2f s  %s" % (nombre, n, n / FPS, "bucle" if loop else ""))

# ---------------------------------------------------------------- salida
ad = arm.animation_data_create()
ad.action = None
for pb in arm.pose.bones:
    pb.matrix_basis.identity()
blend, fbx = out_base + ".blend", out_base + ".fbx"
bpy.ops.export_scene.fbx(
    filepath=fbx, object_types={"ARMATURE", "MESH"}, use_selection=False,
    apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
    add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
    use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
    bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
    bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE",
)
# El .blend se abre con un clip ya cargado: sin Action asignada el editor sale vacio y parece que no hay animaciones.
act = bpy.data.actions["Fly"]
ad.action = act
ad.action_slot = act.slots[0]
scene.frame_start, scene.frame_end = int(act.frame_start), int(act.frame_end)
scene.frame_set(0)
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.file.make_paths_relative()     # la textura queda como //Final/... si esta junto al .blend
bpy.ops.wm.save_mainfile()
print("ANIM_OK", len(CLIPS), blend, fbx)
