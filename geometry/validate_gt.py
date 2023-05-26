import os
import numpy as np
import torch

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from sklearn.decomposition import PCA


gt_offsets = np.load("../polygon_data/output/deform_verts.npz", allow_pickle=True)

all_feats = np.stack([gt_offsets[filename] for filename in gt_offsets.files], axis=0) 
all_feats = all_feats.reshape(all_feats.shape[0], -1)


pca = PCA(n_components=5)
pca.fit(all_feats)


output_dir = "../polygon_data/output/validate_gt_pca"
os.makedirs(output_dir, exist_ok=True)

for filename in gt_offsets.files:
    print(filename)

    # We initialize the source shape to be a sphere of radius 1
    src_mesh = ico_sphere(4)

    offsets = gt_offsets[filename].reshape(-1, 3)
    offsets = pca.inverse_transform(pca.transform(offsets.reshape(-1)[None])).reshape(-1, 3)

    defrom_verts = torch.from_numpy(offsets)

    new_src_mesh = src_mesh.offset_verts(defrom_verts)
    
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    save_obj(os.path.join(output_dir, filename), final_verts, final_faces)

__import__("IPython").embed()
