import sys
import os
import shutil
#package_path
network_yaml = sys.argv[1]
# models = sys.argv[2]
# print(sys.argv)

def get_dir_files(path, postfix):
	'''
	Get files of certain postfix in a directory.
	:param path: the directory to enumerate, absolute path.
	:param postfix: postfix of filename.
	:return: a list of absolute paths.
	'''

	filelist = []
	files = os.listdir(path)
	for item in files:
		if os.path.isfile(os.path.join(path, item)):
			if item.endswith(postfix):
				filelist.append(os.path.join(path, item))
	return filelist

# print(network_yaml, models.split(','))
for model_path in sys.argv[2:]:
	if os.path.isdir(model_path):
		all_files = get_dir_files(model_path, 'pth')

		for file in all_files:
			model_name = os.path.basename(file)
			package_path = os.path.join('./robot_controller/models', model_name[:-4])
			print(package_path)
			os.makedirs(package_path, exist_ok=False)
			model_dst_path = os.path.join(package_path, 'model.pth')
			network_dst_path = os.path.join(package_path, 'network.yaml')

			shutil.copyfile(file, model_dst_path)
			shutil.copyfile(network_yaml, network_dst_path)

	# model_name =
	# package_path = os.path.join('./robot_controller/models', model_name[:-4])
	# print(package_path)
	# os.makedirs(package_path, exist_ok=False)
	# model_dst_path = os.path.join(package_path, 'model.pth')
	# network_dst_path = os.path.join(package_path, 'network.yaml')
	#
	# shutil.copyfile(model_path, model_dst_path)
	# shutil.copyfile(network_yaml, network_dst_path)

	else:
		model_name = os.path.basename(model_path)
		print(model_name)
		package_path = os.path.join('./robot_controller/models', model_name[:-4])
		print(package_path)
		os.makedirs(package_path, exist_ok=False)
		model_dst_path = os.path.join(package_path, 'model.pth')
		network_dst_path = os.path.join(package_path, 'network.yaml')

		shutil.copyfile(model_path, model_dst_path)
		shutil.copyfile(network_yaml, network_dst_path)
