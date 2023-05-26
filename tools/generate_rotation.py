import torch
from numpy.distutils.system_info import gtkp_x11_2_info


def normalize(x, eps: float = 1e-9):
	return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quat_unit(a):
	return normalize(a)

def quat_from_angle_axis(angle, axis):
	theta = (angle / 2).unsqueeze(-1)
	xyz = normalize(axis) * theta.sin()
	w = theta.cos()
	return quat_unit(torch.cat([xyz, w], dim=-1))


PI = 3.1415926
if __name__ == '__main__':
	all_rotations = []
	# x axis rotation
	for theta in [0, PI/2, PI, 3*PI/2]:
		quat = quat_from_angle_axis(torch.tensor([theta]).float().reshape(1), torch.tensor([[1, 0, 0]]).float().reshape(1, 3))
		all_rotations.append(quat)
	#print(all_rotations)

	# y axis rotation
	for theta in [PI/2, PI, 3*PI/2]:
		quat = quat_from_angle_axis(torch.tensor([theta]).float().reshape(1), torch.tensor([[0, 1, 0]]).float().reshape(1, 3))
		all_rotations.append(quat)

	# z axis rotation
	for theta_step in range(0, 16):
		theta = 2 * PI * theta_step / 16
		quat = quat_from_angle_axis(torch.tensor([theta]).float().reshape(1), torch.tensor([[0, 1, 0]]).float().reshape(1, 3))
		all_rotations.append(quat)

	all_rotations = torch.cat(all_rotations)
	print(all_rotations)