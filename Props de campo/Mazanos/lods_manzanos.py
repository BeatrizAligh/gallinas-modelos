"""Poligonaje y LODs de los manzanos (Blender CLI), sobre <carpeta>/ManzanoN_anim.blend (rig + UV + Idle).

blender -b -P lods_manzanos.py               -> los 5 manzanos
blender -b -P lods_manzanos.py -- 1 4        -> sólo esos
blender -b -P lods_manzanos.py -- vistas=dir -> además, renders con textura de cada LOD en dir

Revisión de poligonaje (texturas 2048): los originales tienen 20-23k triángulos. Al 50 % no se distingue
de la original ni de cerca (prueba con textura), así que ése es el LOD0 del juego. Al 22 % sigue igual de
lejos y al 11 % se nota facetado, pero sólo se ve muy lejos. Por debajo del ~8 % la copa se rompe (agujeros).

  _LOD0  50 %  (~10-11.5k tris)  cerca
  _LOD1  22 %  (~4.5-5k)         media distancia
  _LOD2  11 %  (~2.2-2.5k)       lejos y árboles de fondo

La malla original NO se toca y se queda en el .blend (objeto sin sufijo), pero no se exporta: el .fbx lleva el
esqueleto, la animación Idle y los tres LODs. Decimate (colapso) interpola los UV y los pesos, así que los
LODs se mecen igual con el mismo esqueleto. Unity agrupa por los sufijos _LOD0/_LOD1/_LOD2.
"""
import bpy, sys, os, glob, math

BASE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
pedidos = [int(a) for a in argv if a.isdigit()] or [1, 2, 3, 4, 5]
VISTAS = next((a.split("=", 1)[1] for a in argv if a.startswith("vistas=")), None)
NIVELES = [(0, 0.50), (1, 0.22), (2, 0.11)]


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def carpeta(n):
    for d in sorted(glob.glob(os.path.join(BASE, "Manzano*"))):
        if os.path.isdir(d) and "".join(c for c in os.path.basename(d) if c.isdigit()) == str(n):
            return d
    raise SystemExit(f"No encuentro la carpeta del Manzano {n}")


def render_vistas(n, lods, png):
    sc = bpy.context.scene
    mat = bpy.data.materials.new("Vista"); mat.use_nodes = True
    nt = mat.node_tree
    img = nt.nodes.new("ShaderNodeTexImage"); img.image = bpy.data.images.load(png)
    nt.links.new(img.outputs[0], nt.nodes["Principled BSDF"].inputs[0]); nt.nodes.active = img
    shown = []
    for i, o in enumerate(lods):
        c = o.copy(); c.data = o.data.copy(); sc.collection.objects.link(c)
        for m in list(c.modifiers):
            c.modifiers.remove(m)
        c.parent = None; c.matrix_world = o.matrix_world.copy(); c.location.x += i * 1.3
        c.data.materials.clear(); c.data.materials.append(mat)
        shown.append(c)
    hidden = [o for o in sc.objects if o not in shown]
    for o in hidden:
        o.hide_render = True
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x, sc.render.resolution_y = 1500, 560
    cam = bpy.data.objects.new("VistaCam", bpy.data.cameras.new("VistaCam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"; cam.data.ortho_scale = 1.3 * len(shown)
    cam.location = (lods[0].matrix_world.translation.x + 1.3, -6, 0.55); cam.rotation_euler = (math.radians(90), 0, 0)
    os.makedirs(VISTAS, exist_ok=True)
    sc.render.filepath = os.path.join(VISTAS, f"Manzano{n}_lods.png")
    bpy.ops.render.render(write_still=True)
    for o in shown + [cam]:
        bpy.data.objects.remove(o)
    bpy.data.materials.remove(mat)
    for o in hidden:
        o.hide_render = False


for n in pedidos:
    d = carpeta(n)
    blend = glob.glob(os.path.join(d, "*_anim.blend"))[0]
    fbx = blend[:-len(".blend")] + ".fbx"
    bpy.ops.wm.open_mainfile(filepath=blend)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    for o in [o for o in bpy.data.objects if o.type == "MESH" and "_LOD" in o.name]:
        bpy.data.objects.remove(o)
    src = next(o for o in bpy.data.objects if o.type == "MESH")
    assert src.data.uv_layers, f"{src.name} no tiene UV"
    nombre = src.name
    lods, info = [], [("original", tris(src))]
    for lvl, ratio in NIVELES:
        o = src.copy(); o.data = src.data.copy(); o.name = o.data.name = f"{nombre}_LOD{lvl}"
        bpy.context.scene.collection.objects.link(o)
        for m in list(o.modifiers):
            if m.type == "ARMATURE":
                o.modifiers.remove(m)
        dm = o.modifiers.new("Reducir", "DECIMATE")
        dm.decimate_type = "COLLAPSE"; dm.ratio = ratio; dm.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=dm.name)
        a = o.modifiers.new("Armature", "ARMATURE"); a.object = arm
        o.parent = arm
        lods.append(o)
        info.append((f"LOD{lvl}", tris(o)))
    print(f"LODS Manzano {n}: {info}")
    if VISTAS:
        png = sorted(glob.glob(os.path.join(d, "*.png")))[0]
        render_vistas(n, lods, png)
    bpy.ops.wm.save_mainfile()
    if os.path.exists(blend + "1"):
        os.remove(blend + "1")
    # Sólo esqueleto + LODs (la malla original se queda en el .blend como fuente).
    bpy.ops.object.select_all(action="DESELECT")
    for o in [arm] + lods:
        o.select_set(True)
    bpy.ops.export_scene.fbx(
        filepath=fbx, object_types={"ARMATURE", "MESH"}, use_selection=True,
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE")
    print("LODS_OK", fbx)
