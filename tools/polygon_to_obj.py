from shapely import geometry
import trimesh
import json
import os


def read_poly(path):
	with open(path) as json_file:
		poly_dict = json.load(json_file)
	return poly_dict["rect"]


def list_files(checkpoint_path, postfix):
	all_files = os.listdir(checkpoint_path)
	print(all_files)
	file_list = []
	for f in all_files:
		if postfix in f:
			file_list.append(os.path.join(checkpoint_path, f))

	return file_list

all_files = list_files('./rectangle_data', '.txt')

polys = []
for pfile in all_files:
	print(pfile)
	polys += read_poly(pfile)


# we should also move it to the center.
for i, poly in enumerate(polys):
	if len(poly) < 3:
		continue

	polygon_shapely = geometry.Polygon(poly)
	mesh = trimesh.creation.extrude_polygon(polygon_shapely, 2)
	obj_file = mesh.export(file_type='obj')

	with open('./poly_{}.obj'.format(i), 'w') as f:
		f.write(obj_file)
		print("write to {}".format('./poly_{}.obj'.format(i)))