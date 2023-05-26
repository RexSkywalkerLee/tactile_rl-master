import sys
import os
import shutil
from tools.object_convex_decomposition import decompose_obj_mesh

#package_path
asset_root_path = './assets/urdf/objects'
object_dir = 'meshes/set2'
full_source_dir = os.path.join(asset_root_path, object_dir)

with open('./tools/single_collision_template.urdf', 'r') as f:
	single_collision_mesh_urdf = f.read()

with open('./tools/object_template.urdf', 'r') as f:
	template = f.read()


target_object_path = os.path.join(asset_root_path, object_dir)
all_files = os.listdir(target_object_path)
object_names = []

all_files = sorted(all_files)
for f in all_files:
	print(f)

	if '.obj' in f:
		object_name = f[:-4]
		full_object_path = os.path.join(full_source_dir, f)

		decomposed_objs = decompose_obj_mesh(full_object_path)

		collision_text = ''
		for i, obj in enumerate(decomposed_objs):
			collision_mesh_name = object_name + '_' + str(i) + 'decompose.obj'
			with open(os.path.join(full_source_dir, collision_mesh_name), 'w') as f:
				f.write(obj)
			collision_text += single_collision_mesh_urdf.format(obj_file_path=object_dir + '/' + collision_mesh_name)

		new_urdf_file_name = os.path.join(asset_root_path, object_name + '.urdf')
		with open(new_urdf_file_name, 'w') as f:
			f.write(template.format(obj_file_path=object_dir + '/' + object_name + '.obj',
									collision_mesh=collision_text))

		object_names.append(object_name)

		print("\"{obj}\": \"urdf/objects/{obj}.urdf\",".format(obj=object_name))

object_names = sorted(object_names)
print("Get", object_names)
# for object_name in object_names:
# 	new_urdf_file_name = os.path.join(asset_root_path, object_name + '.urdf')
# 	with open(new_urdf_file_name, 'w') as f:
# 		f.write(template.format(obj_file_path=object_dir + '/' + object_name + '.obj'))
# 	print("\"{obj}\": \"urdf/objects/{obj}.urdf\",".format(obj=object_name))
#return os.path.join(checkpoint_path, f)