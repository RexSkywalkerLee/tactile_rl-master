import os

import torch
import copy
import numpy as np
from robot_controller.nn_builder import build_network
from rl_games.algos_torch import torch_ext


class NNAbsoluteLSTMController:
	def __init__(self, dof_lower, dof_upper, num_actors=1,
				 config_path='./robot_controller/network.yaml'):

		self.num_actors = num_actors
		self.model = build_network(config_path)
		self.last_action = torch.zeros(num_actors, 22)
		self.target = torch.zeros(num_actors, 22)
		self.initial_target = torch.zeros(22)

		self.states = None
		self.dof_lower = torch.from_numpy(dof_lower)
		self.dof_upper = torch.from_numpy(dof_upper)

	def reset_rnn(self):
		rnn_states = self.model.get_default_rnn_state()
		self.states = [torch.zeros((s.size()[0], self.num_actors, s.size(
		)[2]), dtype=torch.float32) for s in rnn_states]

	def reset(self):
		self.target = self.initial_target.reshape(-1, 22).repeat(self.num_actors, 1)
		self.reset_rnn()

	def set_initial_target(self, target):
		# target: np.array, absolute qpos.
		self.initial_target = torch.from_numpy(target).float()

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

	def predict(self, observation, deterministic=False):
		'''
			:param observation: np.array, size = (num_actors, 60, ) dtype=np.float32
			:param deterministic: boolean.
			:return: action: np.array, size = (num_actors, 22, )
		'''
		obs = torch.from_numpy(observation).float()
		obs = self._preproc_obs(obs)
		obs[:, 23:45] = self.last_action
		obs[:, 45:60] = torch.zeros_like(obs[:, 45:60])
		input_dict = {
			'is_train': False,
			'prev_actions': None,
			'obs': obs,
			'rnn_states': self.states
		}
		with torch.no_grad():
			res_dict = self.model(input_dict)
		mu = res_dict['mus']
		action = res_dict['actions']
		self.states = res_dict['rnn_states']
		if deterministic:
			current_action = mu
		else:
			current_action = action

		self.last_action = current_action

		current_action = torch.clip(current_action, -1.0, 1.0)

		# Moving average.
		target = self.scale(current_action)
		self.target = 0.1 * target + 0.9 * self.target
		self.target = torch.max(torch.min(self.target, self.dof_upper), self.dof_lower)

		return self.target.detach().cpu().numpy()

	def load(self, fn):
		checkpoint = torch_ext.load_checkpoint(fn)
		self.model.load_state_dict(checkpoint['model'])

		# if self.normalize_input and 'running_mean_std' in checkpoint:
		#	self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


if __name__ == '__main__':
	DOF_LOWER_LIMITS = np.array([0.0, 0.782, -1.4240, 3.286, 2.458,
								 -1.484, -0.47, -0.196, -0.174, -0.227,
								 0.263, -0.105, -0.189, -0.162, -0.47,
								 -0.196, -0.174, -0.227, -0.47, -0.196,
								 -0.174, -0.227])

	# upper limits of each joint
	DOF_UPPER_LIMITS = np.array([0.001, 0.783, -1.4238, 3.287, 2.460,
								 -1.480, 0.47, 1.61, 1.709, 1.618,
								 1.396, 1.163, 1.644, 1.719, 0.47,
								 1.61, 1.709, 1.618, 0.47, 1.61,
								 1.709, 1.618])

	INIT_QPOS = np.array([
		0.00, 0.782, -1.4239, 3.2866, 2.459, -1.4822,
		0.0, 0.0, 0.0, 0.0, 1.3815,
		0.0868, 0.1259, 0.0, 0.0048, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	], dtype=np.float32)

	model_root = './robot_controller/models'
	model_name = 'z_nov22_lstm'
	model_path = os.path.join(model_root, model_name)

	network_config_path = os.path.join(model_path, 'network.yaml')
	network_checkpoint_path = os.path.join(model_path, 'model.pth')
	policy = NNAbsoluteLSTMController(dof_lower=DOF_LOWER_LIMITS, dof_upper=DOF_UPPER_LIMITS, num_actors=1,
						  		  	  config_path=network_config_path)

	policy.load(network_checkpoint_path)
	policy.set_initial_target(INIT_QPOS)
	policy.reset()

	for i in range(100):
		obs = np.random.uniform(0, 1, 60)

		''' 
			Observation description: 
			
			DIMENSION |	DESCRIPTION	 |	PROGRAMMING HINT
			-------------------------------------------------------------------------------------------
			[0:22]:   |	current_qpos |	(SET BY USER. 			Set it by the previous protocol)
			[22:23]:  |	0  		   	 |	(AUTOSET BY CONTROLLER. Leave it blank)
			[23:45]:  |	last_action  |	(AUTOSET BY CONTROLLER. Leave it blank)
			[45:60]:  |	sensor_obs   |	(SET BY USER. 			Set it to 0 now)
		'''

		action = policy.predict(observation=obs.reshape(1, -1))[0]

		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
			
			For relative control (This version), the update is: qpos_target = qpos + action * scale.
			For absolute control, the update is: qpos_target = rescale(action). The action is internally EMAed.	
		'''

		print(action)
