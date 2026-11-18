"""Inspección de los props nuevos (0.9.4): tris, tamaño, UVs, islas, materiales.
Uso: blender -b --factory-startup -P inspeccionar_props.py -- <salida.json> <fbx1> <fbx2> ...
"""
import bpy, bmesh, sys, json, os

args = sys.argv[sys.argv.index("--") + 1:]
out, files = args[0], args[1:]
res = {}
for f in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=f)
    info = {"objetos": []}
    mins = [1e9] * 3; maxs = [-1e9] * 3
    for o in bpy.context.scene.objects:
        e = {"nombre": o.name, "tipo": o.type, "esc": list(o.scale), "rot": list(o.rotation_euler)}
        if o.type == "MESH":
            me = o.data
            bm = bmesh.new(); bm.from_mesh(me)
            tris = sum(len(fc.verts) - 2 for fc in bm.faces)
            # islas por conectividad de vértices
            seen = set(); islas = 0
            for v in bm.verts:
                if v.index in seen: continue
                islas += 1
                st = [v]
                while st:
                    x = st.pop()
                    if x.index in seen: continue
                    seen.add(x.index)
                    for ed in x.link_edges:
                        st.append(ed.other_vert(x))
            abiertos = sum(1 for ed in bm.edges if len(ed.link_faces) == 1)
            bm.free()
            for c in o.bound_box:
                w = o.matrix_world @ __import__("mathutils").Vector(c)
                for i in range(3):
                    mins[i] = min(mins[i], w[i]); maxs[i] = max(maxs[i], w[i])
            e.update(tris=tris, verts=len(me.vertices), uvs=[u.name for u in me.uv_layers], islas=islas,
                     bordes_abiertos=abiertos, materiales=[m.name if m else None for m in me.materials])
        info["objetos"].append(e)
    info["tam_mundo"] = [round(maxs[i] - mins[i], 3) for i in range(3)]
    info["min"] = [round(x, 3) for x in mins]
    info["imagenes"] = [i.filepath for i in bpy.data.images]
    res[os.path.basename(f)] = info
json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("OK", len(res))
