import torch
import copy
import numpy as np
from robot_controller.nn_builder import build_network
from rl_games.algos_torch import torch_ext

class NNController:
	def __init__(self, num_actors=1, config_path='./robot_controller/network.yaml'):
		self.model = build_network(config_path)
		self.last_action = torch.zeros(num_actors, 22)
		self.states = None

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
		return current_action.detach().cpu().numpy()

	def load(self, fn):
		checkpoint = torch_ext.load_checkpoint(fn)
		self.model.load_state_dict(checkpoint['model'])

		# if self.normalize_input and 'running_mean_std' in checkpoint:
		#	self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


if __name__ == '__main__':
	policy = NNController(num_actors=1,
						  config_path='./robot_controller/network.yaml')

	policy.load('./robot_controller/models/y_slow_rel.pth')

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
