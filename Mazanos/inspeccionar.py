"""Inspecciona los FBX de los manzanos: objetos, materiales, islas y medidas. blender -b -P inspeccionar.py"""
import bpy, bmesh, os, glob
from mathutils import Vector

DIR = os.path.dirname(os.path.abspath(__file__))
for path in sorted(glob.glob(os.path.join(DIR, "*.fbx"))):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    print("=" * 70)
    print(os.path.basename(path))
    for o in bpy.data.objects:
        line = f"  {o.type:8} {o.name!r} parent={o.parent.name if o.parent else None} loc={tuple(round(v,3) for v in o.location)} scale={tuple(round(v,3) for v in o.scale)} rot={tuple(round(v,2) for v in o.rotation_euler)}"
        print(line)
        if o.type != 'MESH':
            continue
        me = o.data
        ws = [o.matrix_world @ v.co for v in me.vertices]
        mn = Vector((min(v.x for v in ws), min(v.y for v in ws), min(v.z for v in ws)))
        mx = Vector((max(v.x for v in ws), max(v.y for v in ws), max(v.z for v in ws)))
        print(f"     verts={len(me.vertices)} faces={len(me.polygons)} uv={[u.name for u in me.uv_layers]} col={[c.name for c in me.color_attributes]}")
        print(f"     bbox min={tuple(round(v,2) for v in mn)} max={tuple(round(v,2) for v in mx)}")
        mats = {}
        for p in me.polygons:
            mats[p.material_index] = mats.get(p.material_index, 0) + 1
        for i, n in sorted(mats.items()):
            m = o.material_slots[i].material if i < len(o.material_slots) else None
            tex = []
            if m and m.use_nodes:
                tex = [nd.image.name for nd in m.node_tree.nodes if nd.type == 'TEX_IMAGE' and nd.image]
            print(f"     mat[{i}] {m.name if m else None}: {n} caras tex={tex}")
        # islas por material
        bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        seen = set(); islands = []
        for f in bm.faces:
            if f.index in seen: continue
            stack = [f]; seen.add(f.index); isl = []
            while stack:
                g = stack.pop(); isl.append(g)
                for e in g.edges:
                    for h in e.link_faces:
                        if h.index not in seen:
                            seen.add(h.index); stack.append(h)
            islands.append(isl)
        print(f"     islas={len(islands)}")
        summary = {}
        for isl in islands:
            mi = isl[0].material_index
            zs = [(o.matrix_world @ v.co).z for f in isl for v in f.verts]
            summary.setdefault(mi, []).append((len(isl), round(min(zs), 2), round(max(zs), 2)))
        for mi, lst in summary.items():
            lst.sort(reverse=True)
            print(f"     mat{mi}: {len(lst)} islas, mayores {lst[:6]}")
        nonman = sum(1 for e in bm.edges if not e.is_manifold)
        print(f"     aristas no-manifold={nonman}")
        bm.free()
