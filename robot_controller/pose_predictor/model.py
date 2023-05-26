import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNStatePosePredictor(nn.Module):
	def __init__(self):
		super(RNNStatePosePredictor, self).__init__()
		self.network = nn.Sequential(nn.Linear(1024, 256),
									 nn.ReLU(),
									 nn.Linear(256, 256),
									 nn.ReLU(),
									 nn.Linear(256, 4))

	def forward(self, x):
		# We predict the increment of the quaternion of two consecutive frames
		return self.network(x)

	def loss(self, x, gt_y):
		return F.mse_loss(self.network(x), gt_y)

