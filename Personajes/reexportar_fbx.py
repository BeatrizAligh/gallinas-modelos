"""Vuelve a exportar el FBX de un rig ya hecho: blender -b <Nombre>_rig.blend -P reexportar_fbx.py
Cada acción sale como una toma (como el gavilán)."""
import bpy, os
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
body = [o for o in bpy.data.objects if o.type == 'MESH'][0]
arm.animation_data.action = None
for b in arm.pose.bones:
    b.matrix_basis.identity()
out = os.path.join(os.path.dirname(bpy.data.filepath), body.name + '.fbx')
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True); arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_scene.fbx(filepath=out, use_selection=True, object_types={'ARMATURE', 'MESH'},
                         apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
                         add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
                         bake_anim_use_all_bones=True, bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
                         primary_bone_axis='Y', secondary_bone_axis='X', mesh_smooth_type='FACE')
print('REEXPORT', out, len(bpy.data.actions))
