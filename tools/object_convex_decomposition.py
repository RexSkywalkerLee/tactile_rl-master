import trimesh
import trimesh.exchange.obj as tobj
import trimesh.decomposition as decomposite
import os

# def as_mesh(x):
# 	if isinstance(x, trimesh.Scene):
# 		if len(x.geometry) == 0:
# 			mesh = None
# 		else:
# 			mesh = trimesh.util.concatenate(tuple(
# 				trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in x.geometry.values()
# 			))
# 	else:
# 		mesh = x
# 	return mesh


def decompose_obj_mesh(obj_path):
	obj_mesh = trimesh.load(obj_path)
	meshes = decomposite.convex_decomposition(obj_mesh)
	# obj_name = os.path.basename(obj_path)[:-4]
	print(meshes)
	if isinstance(meshes, list):
		return [mesh.export(file_type='obj') for mesh in meshes]
	elif isinstance(meshes, trimesh.Trimesh):
		return [meshes.export(file_type='obj')]
	assert False

	#print(obj_kwargs)


if __name__ == '__main__':
	decompose_obj_mesh('../assets/urdf/objects/meshes/set2/set_obj14_irregular_block_cross.obj')
