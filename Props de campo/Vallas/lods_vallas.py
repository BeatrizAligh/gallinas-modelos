"""Poligonaje y LODs de las vallas del corral (Blender CLI).

blender -b --factory-startup -P lods_vallas.py                -> Vallas_LOD.blend + Vallas_LOD.fbx
blender -b --factory-startup -P lods_vallas.py -- vistas=dir  -> ademas, renders con textura de cada LOD en dir

Revision de poligonaje (texturas 2048): las piezas originales tienen 18-25k triangulos cada una y el corral usa
~16 tramos y ~20 postes (~800k triangulos solo de valla). Son formas redondeadas sin detalle fino en la malla
(las vetas estan en la textura), asi que aguantan mucha reduccion:

  _LOD0  3000 triangulos  cerca           (no se distingue de la original)
  _LOD1  1200 (poste 1500)  media distancia
  _LOD2   450 (poste  900)  lejos

Los FBX originales NO se tocan. Cada pieza sale con el origen en el centro de su base, z arriba, y el largo de la
valla sobre X (el poste lleva los tacos de union tambien sobre X). Unity coge las mallas por nombre (<Pieza>_LODn).
"""
import bpy, sys, os, math

BASE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
VISTAS = next((a.split("=", 1)[1] for a in argv if a.startswith("vistas=")), None)
PIEZAS = [("Poste_Valla", "Poste_Valla.fbx", "Poste_Valla.png"),
          ("Valla_Buena", "Valla_Buena.fbx", "V_Buena.png"),
          ("Valla_Medio_Rota", "Valla_medio_rota.fbx", "V_Medio_Rota.png"),
          ("Valla_Rota", "Valla_Rota.fbx", "V_Rota.png")]
OBJETIVO = [3000, 1200, 450]
# El poste es un cilindro con tacos y tornillos: por debajo de ~900 se le abren agujeros en los tornillos.
OBJETIVO_PIEZA = {"Poste_Valla": [3000, 1500, 900]}


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
todas, info = [], []
for nombre, fbx, png in PIEZAS:
    antes = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=os.path.join(BASE, fbx))
    nuevos = [o for o in bpy.data.objects if o not in antes]
    src = next(o for o in nuevos if o.type == "MESH")
    for o in nuevos:
        if o != src:
            bpy.data.objects.remove(o)
    assert src.data.uv_layers, f"{nombre} no tiene UV"
    # Transformacion aplicada: z arriba, origen en el centro de la base.
    src.data.transform(src.matrix_world)
    src.matrix_world.identity()
    xs = [v.co.x for v in src.data.vertices]; ys = [v.co.y for v in src.data.vertices]; zs = [v.co.z for v in src.data.vertices]
    from mathutils import Matrix
    src.data.transform(Matrix.Translation((-(min(xs) + max(xs)) / 2, -(min(ys) + max(ys)) / 2, -min(zs))))
    src.name = src.data.name = nombre
    img = bpy.data.images.load(os.path.join(BASE, "Textura", png), check_existing=True)
    mat = bpy.data.materials.new(nombre); mat.use_nodes = True
    nt = mat.node_tree
    ti = nt.nodes.new("ShaderNodeTexImage"); ti.image = img
    nt.links.new(ti.outputs["Color"], next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"])
    src.data.materials.clear(); src.data.materials.append(mat)
    for p in src.data.polygons:
        p.use_smooth = True
    # Caras internas (el poste trae ~900 donde los tacos entran en el tronco): no se ven y bloquean la reduccion.
    import bmesh
    bm = bmesh.new(); bm.from_mesh(src.data)
    internas = [f for f in bm.faces if all(len(e.link_faces) > 2 for e in f.edges)]
    if internas:
        bmesh.ops.delete(bm, geom=internas, context="FACES")
        bm.to_mesh(src.data)
    bm.free()
    t0 = tris(src)
    fila = [nombre, t0, len(internas), tuple(round(x, 3) for x in (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))]
    for lvl, obj in enumerate(OBJETIVO_PIEZA.get(nombre, OBJETIVO)):
        o = src.copy(); o.data = src.data.copy(); o.name = o.data.name = f"{nombre}_LOD{lvl}"
        sc.collection.objects.link(o)
        dm = o.modifiers.new("Reducir", "DECIMATE")
        dm.decimate_type = "COLLAPSE"; dm.ratio = min(1.0, obj / t0); dm.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=dm.name)
        todas.append(o)
        fila.append(tris(o))
    sc.collection.objects.link(src) if src.name not in sc.collection.objects else None
    src.hide_set(True); src.hide_render = True
    info.append(fila)
    print("LODS", fila)


def render_vistas():
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x, sc.render.resolution_y = 1800, 900
    cam = bpy.data.objects.new("VistaCam", bpy.data.cameras.new("VistaCam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"
    os.makedirs(VISTAS, exist_ok=True)
    for nombre, _, _ in PIEZAS:
        src = bpy.data.objects[nombre]
        grupo = [src] + [bpy.data.objects[f"{nombre}_LOD{i}"] for i in range(len(OBJETIVO))]
        ancho = max(src.dimensions.x, src.dimensions.y) + 0.25
        guardado = {o: o.location.copy() for o in grupo}
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.hide_render = o not in grupo
        for i, o in enumerate(grupo):
            o.location.x = i * ancho
            o.hide_set(False)
        cam.data.ortho_scale = ancho * len(grupo) * 1.02
        d = (0.45, -1, 0.55)
        n = math.sqrt(sum(x * x for x in d)); d = tuple(x / n for x in d)
        from mathutils import Vector
        tgt = Vector((ancho * (len(grupo) - 1) / 2, 0, src.dimensions.z / 2))
        cam.location = tgt + Vector(d) * 8
        cam.rotation_euler = (-Vector(d)).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(VISTAS, f"{nombre}_lods.png")
        bpy.ops.render.render(write_still=True)
        for o, p in guardado.items():
            o.location = p
    bpy.data.objects.remove(cam)


if VISTAS:
    render_vistas()
for nombre, _, _ in PIEZAS:
    bpy.data.objects[nombre].hide_render = True
blend = os.path.join(BASE, "Vallas_LOD.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_mainfile()
if os.path.exists(blend + "1"):
    os.remove(blend + "1")
# Solo los LODs: los originales se quedan en el .blend como referencia.
bpy.ops.object.select_all(action="DESELECT")
for o in todas:
    o.hide_set(False)
    o.select_set(True)
bpy.ops.export_scene.fbx(
    filepath=os.path.join(BASE, "Vallas_LOD.fbx"), object_types={"MESH"}, use_selection=True,
    apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
    use_mesh_modifiers=False, mesh_smooth_type="FACE", bake_anim=False, path_mode="RELATIVE")
print("RESUMEN (pieza, tris original, caras internas quitadas, medidas, LOD0, LOD1, LOD2)")
for f in info:
    print("  ", f)
print("LODS_OK")
