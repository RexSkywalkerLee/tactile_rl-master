import os
import torch
import pytorch3d

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
import numpy as np
from tqdm import tqdm

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

output_dir = "../polygon_data/output"
mesh_dir = "../polygon_data/meshes"
os.makedirs(output_dir, exist_ok=True)


def generate_gt(mesh_filename):
    trg_obj = os.path.join(mesh_dir, mesh_filename)

    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    # center = verts.mean(0)
    # verts = verts - center
    # scale = max(verts.abs().max(0)[0])
    # verts = verts / scale

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

    # We initialize the source shape to be a sphere of radius 1
    src_mesh = ico_sphere(4, device)

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    # Number of optimization steps
    Niter = 200
    # Weight for the chamfer loss
    w_chamfer = 1.0
    # Weight for mesh edge loss
    w_edge = 1.0
    # Weight for mesh normal consistency
    w_normal = 0.01
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1
    # Plot period for the losses
    plot_period = 250
    loop = tqdm(range(Niter))

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)

        # We sample 5k points from the surface of each mesh
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)

        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

        # Print the losses
        loop.set_description("total_loss = %.6f" % loss)

        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))

        # Optimization step
        loss.backward()
        optimizer.step()

    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # Scale normalize back to the original target size
    # final_verts = final_verts * scale + center

    # Store the predicted mesh using save_obj
    final_obj = os.path.join(output_dir, mesh_filename)
    save_obj(final_obj, final_verts, final_faces)

    return deform_verts.detach().cpu().numpy()


mesh_filenames = sorted(os.listdir(mesh_dir))

# for mesh_filename in mesh_filenames:
#     all_verts = []
#     if mesh_filename.endswith(".obj"):
#         verts, faces, aux = load_obj(os.path.join(mesh_dir, mesh_filename))
#         all_verts.append(verts)

#     all_verts = torch.cat(all_verts, dim=0)
#     center = all_verts.mean(0)
#     all_verts = all_verts - center
#     scale = max(all_verts.abs().max(0)[0])
#     all_verts = all_verts / scale


result = {}
for mesh_filename in mesh_filenames:
    print(mesh_filename)
    if mesh_filename.endswith(".obj"):
        result[mesh_filename] = generate_gt(mesh_filename)

np.savez_compressed(os.path.join(output_dir, "deform_verts.npz"), **result)

__import__("IPython").embed()
