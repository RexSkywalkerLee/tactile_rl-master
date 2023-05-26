import sys
import os
import shutil

import numpy as np
import trimesh
import trimesh.decomposition as decomposite
import trimesh.bounds as bounds
#package_path
# asset_root_path = './assets/urdf/objects'
# object_dir = 'meshes/set2-raw'
# #object_root_path_yaml = sys.argv[1]
# # models = sys.argv[2]
# # print(sys.argv)
#
# with open('./tools/object_raw_template.urdf', 'r') as f:
# 	template = f.read()
#
# target_object_path = os.path.join(asset_root_path, object_dir)
#
# all_files = os.listdir(target_object_path)
#
# object_names = []
# object_names_python = []
#
# for f in all_files:
# 	if '.obj' in f:
# 		object_name = f[:-4]
# 		object_names.append(object_name)
# 		object_names_python.append(object_name + '_raw')
#
# object_names = sorted(object_names)
# object_names_python = sorted(object_names_python)
#
# print("Get", object_names_python)
# for object_name in object_names:
# 	new_urdf_file_name = os.path.join(asset_root_path, object_name + '_raw.urdf')
# 	with open(new_urdf_file_name, 'w') as f:
# 		f.write(template.format(obj_file_path=object_dir + '/' + object_name + '.obj'))
# 	print("\"{obj}_raw\": \"urdf/objects/{obj}_raw.urdf\",".format(obj=object_name))
# #return os.path.join(checkpoint_path, f)

def rescale_obj_mesh(obj_path):
	obj_mesh = trimesh.load(obj_path)
	to_origin, extends = bounds.oriented_bounds(obj_mesh)
	obj_mesh.apply_transform(to_origin)

	to_origin, extends = bounds.oriented_bounds(obj_mesh)
	desired_scale = 2

	scale_transform = np.eye(4)
	scale_transform[0, 0] = desired_scale / extends[0]
	scale_transform[1, 1] = desired_scale / extends[1]
	scale_transform[2, 2] = desired_scale / extends[2]
	# print(to_origin, extends)
	obj_mesh.apply_transform(scale_transform)
	return obj_mesh.export(file_type='obj')

	# meshes = decomposite.convex_decomposition(obj_mesh)
	# obj_name = os.path.basename(obj_path)[:-4]
	# print(meshes)
	# if isinstance(meshes, list):
	# 	return [mesh.export(file_type='obj') for mesh in meshes]
	# elif isinstance(meshes, trimesh.Trimesh):
	# 	return [meshes.export(file_type='obj')]
	# assert False

	#print(obj_kwargs)


if __name__ == '__main__':
	#rescale_obj_mesh('../assets/urdf/objects/meshes/subset_100/B06_0.obj')
	# package_path
	asset_root_path = '../assets/urdf/objects'
	object_dir = 'meshes/polygon'
	output_object_dir = 'meshes/polygon_scaled'

	full_source_dir = os.path.join(asset_root_path, object_dir)
	full_target_dir = os.path.join(asset_root_path, output_object_dir)
	all_files = os.listdir(full_source_dir)

	for f in all_files:
		print(f)

		if '.obj' in f:
			object_name = f[:-4]
			full_object_path = os.path.join(full_source_dir, f)

			rescaled_obj = rescale_obj_mesh(full_object_path)

			rescaled_mesh_name = object_name + '_s.obj'

			with open(os.path.join(full_target_dir, rescaled_mesh_name), 'w') as f:
				f.write(rescaled_obj)

