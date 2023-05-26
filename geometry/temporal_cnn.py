import torch
import torch.nn as nn

class TemporalConv(nn.Module):
	def __init__(self):
		super().__init__()
		self.mlp1 = nn.Sequential(
			nn.Conv1d(85, 64, 5, 2, padding=1),
			nn.ReLU(),
			nn.Conv1d(64, 128, 5, 2, padding=1),
			nn.ReLU(),
			nn.Conv1d(128, 256, 5, 2, padding=1),
			nn.ReLU(),
			nn.Conv1d(256, 256, 5, 2, padding=1),
			nn.ReLU(),
			nn.Conv1d(256, 256, 5, 2, padding=1)
		)

		self.mlp2 = nn.Sequential(nn.Linear(256 * 5, 128),
								  nn.ReLU(),
								  nn.Linear(128, 126))

		self.mlp3 = nn.Sequential(nn.Linear(256 * 5, 128),
								  nn.ReLU(),
								  nn.Linear(128, 126))
		# self.mlp2 = nn.Sequential(
		#     nn.Linear(256, 128),
		#     nn.ReLU(),
		#     nn.Linear(128, 126),
		# )
		#
		# self.mlp3 = nn.Sequential(
		#     nn.Linear(256, 128),
		#     nn.ReLU(),
		#     nn.Linear(128, args.regress_num),
		# )

	def forward(self, x):
		#x = x.reshape(x.size(0), -1, self.stack_num * 85)
		#x = x.permute(0, 2, 1)
		x = self.mlp1(x)
		x = x.reshape(x.size(0), -1)
		y1 = self.mlp1(x)
		y2 = self.mlp2(x)
		print(x.shape)
		# x = x.mean(dim=1)
		# y1 = self.mlp2(x)
		# y2 = self.mlp3(x)
		# return y1, y2

if __name__ == '__main__':
	m = TemporalConv()
	m(torch.randn(12, 85, 200))
