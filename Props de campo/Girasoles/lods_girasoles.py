"""LODs de los girasoles (Blender CLI), sobre los Animados/Girasol N_anim.blend que genera rig_girasoles.py.

blender -b -P lods_girasoles.py -- [1 2 3 4 5]

La malla original queda intacta como _LOD0. _LOD1 (25 %) y _LOD2 (7 %) son copias reducidas con Decimate
(colapso) que heredan los pesos interpolados y van al mismo esqueleto, asi que se mecen igual.
Unity crea el LODGroup solo con los sufijos _LOD0/_LOD1/_LOD2. Se reescribe el .fbx (el .blend se guarda con los LODs).
"""
import bpy, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
flores = [int(a) for a in argv if a.isdigit()] or [1, 2, 3, 4, 5]
NIVELES = [(1, 0.25), (2, 0.07)]


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


for n in flores:
    blend = f"{BASE}/Animados/Girasol {n}_anim.blend"
    bpy.ops.wm.open_mainfile(filepath=blend)
    arm = bpy.data.objects["Girasol_Rig"]
    base = [o for o in bpy.data.objects if o.type == "MESH" and "_LOD" not in o.name]
    for o in [o for o in bpy.data.objects if o.type == "MESH" and "_LOD" in o.name and not o.name.endswith("_LOD0")]:
        bpy.data.objects.remove(o)
    src = base[0] if base else bpy.data.objects[f"Girasol_{n}_LOD0"]
    src.name = f"Girasol_{n}_LOD0"
    info = [("LOD0", tris(src))]
    for lvl, ratio in NIVELES:
        o = src.copy(); o.data = src.data.copy(); o.name = o.data.name = f"Girasol_{n}_LOD{lvl}"
        bpy.context.scene.collection.objects.link(o)
        for m in list(o.modifiers):
            if m.type == "ARMATURE":
                o.modifiers.remove(m)
        d = o.modifiers.new("Reducir", "DECIMATE")
        d.decimate_type = "COLLAPSE"; d.ratio = ratio; d.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=d.name)
        a = o.modifiers.new("Armature", "ARMATURE"); a.object = arm
        o.parent = arm
        info.append((f"LOD{lvl}", tris(o)))
    print(f"LODS Girasol {n}: {info}")
    bpy.ops.wm.save_mainfile()
    if os.path.exists(blend + "1"):
        os.remove(blend + "1")
    bpy.ops.export_scene.fbx(
        filepath=f"{BASE}/Animados/Girasol {n}_anim.fbx", object_types={"ARMATURE", "MESH"}, use_selection=False,
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        use_armature_deform_only=False, use_mesh_modifiers=False, mesh_smooth_type="FACE",
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="RELATIVE")
