import sys
import os
import shutil

#package_path
asset_root_path = './assets/urdf/objects'
object_dir = 'meshes/set2-raw'
#object_root_path_yaml = sys.argv[1]
# models = sys.argv[2]
# print(sys.argv)

with open('./tools/object_raw_template.urdf', 'r') as f:
	template = f.read()

target_object_path = os.path.join(asset_root_path, object_dir)

all_files = os.listdir(target_object_path)

object_names = []
object_names_python = []

for f in all_files:
	if '.obj' in f:
		object_name = f[:-4]
		object_names.append(object_name)
		object_names_python.append(object_name + '_raw')

object_names = sorted(object_names)
object_names_python = sorted(object_names_python)

print("Get", object_names_python)
for object_name in object_names:
	new_urdf_file_name = os.path.join(asset_root_path, object_name + '_raw.urdf')
	with open(new_urdf_file_name, 'w') as f:
		f.write(template.format(obj_file_path=object_dir + '/' + object_name + '.obj'))
	print("\"{obj}_raw\": \"urdf/objects/{obj}_raw.urdf\",".format(obj=object_name))
#return os.path.join(checkpoint_path, f)
