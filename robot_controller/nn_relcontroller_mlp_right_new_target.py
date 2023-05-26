import os

import torch
import copy
import numpy as np
from robot_controller.nn_builder import build_network, get_model
from rl_games.algos_torch import torch_ext


class NNRelativeMLPControllerMP:
	def __init__(self, dof_lower, dof_upper, num_actors=1, scale=0.5, stack=4,
				 config_path='./robot_controller/network.yaml', device='cuda'):

		self.num_actors = num_actors
		self.model = build_network(config_path, input_shape=85 * stack, normalize=True).to(device)
		self.model.eval()
		self.last_action = torch.zeros(num_actors, 22).to(device)
		self.prev_target = torch.zeros(num_actors, 22).to(device)
		self.initial_target = torch.zeros(22).to(device)

		self.states = None
		self.dof_lower = torch.from_numpy(dof_lower).to(device)
		self.dof_upper = torch.from_numpy(dof_upper).to(device)
		self.all_commands = torch.tensor([[1.0, 0.0, 0.0],
										  [0.0, 1.0, 0.0],
										  [0.0, 0.0, 1.0]]).float()
		self.current_cmd = 0
		self.num_supported_cmd = 3

		self.need_init = 0
		self.last_action = torch.zeros(num_actors, 22).to(device)
		self.m_scale = scale

		self.single_step_obs_dim = 85
		self.stack = stack
		self.history = torch.zeros(num_actors, self.single_step_obs_dim * self.stack).float()
		self.device = 'cuda'

	# def reset_rnn(self):
	# 	rnn_states = self.model.get_default_rnn_state()
	# 	self.states = [torch.zeros((s.size()[0], self.num_actors, s.size(
	# 	)[2]), dtype=torch.float32) for s in rnn_states]
	def unnormalize(self, qpos):
		return (0.5 * (qpos + 1.0) * (self.dof_upper - self.dof_lower) + self.dof_lower)

	def normalize(self, qpos):
		return (2.0 * qpos - self.dof_upper - self.dof_lower) / (self.dof_upper - self.dof_lower)

	def reset(self):
		self.prev_target = self.initial_target.reshape(-1, 22).repeat(self.num_actors, 1)
		self.history = torch.zeros(self.num_actors, self.single_step_obs_dim * self.stack).float()
		self.need_init = 1
		# self.reset_rnn()

	def set_initial_target(self, target):
		# target: np.array, absolute qpos.
		self.initial_target = torch.from_numpy(target).float().to(self.device)

	def scale(self, actions):
		return (0.5 * (actions + 1.0) * (self.dof_upper - self.dof_lower) + self.dof_lower)

	def _preproc_obs(self, obs_batch):
		if type(obs_batch) is dict:
			obs_batch = copy.copy(obs_batch)
			for k, v in obs_batch.items():
				if v.dtype == torch.uint8:
					obs_batch[k] = v.float() / 255.0
				else:
					obs_batch[k] = v
		else:
			if obs_batch.dtype == torch.uint8:
				obs_batch = obs_batch.float() / 255.0
		return obs_batch

	def set_command(self, command_id):
		'''

		:param command_id: an integer,
		:return:
		'''
		assert command_id < self.num_supported_cmd, "command is not supported"
		self.current_cmd = command_id

	def get_command(self, batchsize):
		x = self.all_commands[self.current_cmd].reshape(1, -1).repeat(batchsize, 8)
		# print(x.shape)
		return x

	def predict(self, observation, deterministic=False):
		'''
			:param observation: np.array, size = (num_actors, 84, ) dtype=np.float32
			:param deterministic: boolean.
			:return: action: np.array, size = (num_actors, 22, )
		'''
		obs = torch.from_numpy(observation).float().to(self.device)
		obs = self._preproc_obs(obs)
		# print(obs)
		obs[:, :6] = 0
		obs[:, 22:29] = 0
		obs[:, 23+6:45] = self.normalize(self.prev_target)[:, 6:22] #self.last_action
		obs[:, 45:61] = torch.zeros_like(obs[:, 45:61])
		obs[:, 61:85] = self.get_command(obs.size(0))

		if self.need_init:
			self.history = obs.repeat(1, self.stack)
			self.need_init = 0

		self.history = torch.cat((obs, self.history[:, :-self.single_step_obs_dim]), dim=-1)
		input_dict = {
			'is_train': False,
			'prev_actions': None,
			'obs': self.history,
			'rnn_states': None, #self.states
		}
		with torch.no_grad():
			res_dict = self.model(input_dict)
		mu = res_dict['mus']
		action = res_dict['actions']
		# self.states = res_dict['rnn_states']
		if deterministic:
			current_action = mu
		else:
			current_action = action

		current_action = torch.clip(current_action, -1.0, 1.0)
		self.last_action = self.last_action * 0.2 + current_action * 0.8
		self.last_action = current_action

		q_pos_delta = self.m_scale * self.last_action
		self.prev_target = torch.clamp(q_pos_delta + self.prev_target, self.dof_lower, self.dof_upper)

		# Moving average.
		#target = self.scale(current_action)
		#self.target = 0.1 * target + 0.9 * self.target
		#self.target = torch.max(torch.min(self.target, self.dof_upper), self.dof_lower)

		return q_pos_delta.detach().cpu().numpy() #self.target.detach().cpu().numpy()

	def load(self, fn):
		checkpoint = torch_ext.load_checkpoint(fn)
		self.model.load_state_dict(checkpoint['model'])
		#if 'running_mean_std' in checkpoint:
			#print("Loaded")

		self.model.running_mean_std.count.data = (
			checkpoint['model']['running_mean_std.count'].data).to(self.device)
		self.model.running_mean_std.running_mean.data = (
			checkpoint['model']['running_mean_std.running_mean'].data).to(self.device)
		self.model.running_mean_std.running_var.data = (
			checkpoint['model']['running_mean_std.running_var'].data).to(self.device)
		print(checkpoint['model']['running_mean_std.running_var'].data)
		return (checkpoint['model']['running_mean_std.running_mean'].data).detach().cpu().numpy()


if __name__ == '__main__':
	DOF_LOWER_LIMITS = np.array( [0.0, 0.782, -1.42391, 3.2866, 2.459, -1.48221, -0.47, -0.196, -0.174, -0.227, 0.7,
								  0.3, -0.189, -0.162, -0.47, -0.196, -0.174, -0.227, -0.47, -0.196, -0.174, -0.227])

	# upper limits of each joint
	DOF_UPPER_LIMITS = np.array([1e-05, 0.7820001, -1.4239, 3.28661, 2.4591, -1.4822, 0.47, 1.61, 1.709, 1.618,
								 1.396, 1.163, 1.644, 1.719, 0.47, 1.61, 1.709, 1.618, 0.47, 1.61, 1.709, 1.618])

	INIT_QPOS = np.array([
		0.00, 0.782, -1.4239, 3.2866, 2.459, -1.4822,
		0.0, 0.0, 0.0, 0.0, 1.3815,
		0.0868, 0.1259, 0.0, 0.0048, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	], dtype=np.float32)

	model_root = './robot_controller/models'
	model_name = 'jan27_test'
	model_path = os.path.join(model_root, model_name)

	network_config_path = os.path.join(model_path, 'network.yaml')
	network_checkpoint_path = get_model(model_path)
	policy = NNRelativeMLPControllerMP(dof_lower=DOF_LOWER_LIMITS, dof_upper=DOF_UPPER_LIMITS, num_actors=1,
						  		  	  config_path=network_config_path, scale=0.2)

	policy.load(network_checkpoint_path)
	policy.set_initial_target(INIT_QPOS)
	policy.reset()

	'''
		Dec 13:
		This one is new. 
		You can set command to 0(x-axis) or 1(y-axis) or 2(z-axis) in the execution loop.
	'''
	policy.set_command(2)

	for i in range(100):
		obs = np.random.randn(85)
		''' 
			Observation description: 
			
			DIMENSION |	DESCRIPTION	 |	PROGRAMMING HINT
			-------------------------------------------------------------------------------------------
			[0:22]:   |	current_qpos |	(SET BY USER. 			Set it by the previous protocol)
			[22:23]:  |	0  		   	 |	(AUTOSET BY CONTROLLER. Leave it blank)
			[23:45]:  |	last_action  |	(AUTOSET BY CONTROLLER. Leave it blank)
			[45:61]:  |	sensor_obs   |	(SET BY USER. 			Contact Binaries)
			[61:85]:  | command	     |  (AUTOSET BY CONTROLLER. Leave it blank)
		'''
		#obs[22:85] = 0.0

		action = policy.predict(observation=obs.reshape(1, -1))[0]
		# TODO(): qpos = qpos + action
		# TODO(): set_qpos(qpos)
		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
			
			For relative control (This version), the update is: qpos_target = qpos + action * scale.
			For absolute control, the update is: qpos_target = rescale(action). The action is internally EMAed.	
		'''

		print(action)
