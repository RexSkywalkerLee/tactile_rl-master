import sys
import os

import torch
import copy
import numpy as np
import pytorch_kinematics as pk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_controller.nn_builder import build_network, get_model
from rl_games.algos_torch import torch_ext


def get_sensor_pos(joint_angles, urdf_path='./assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin.urdf'):
	batch_size = joint_angles.size(0)
	sensor_pos = torch.zeros((batch_size, 3, 16))

	default_arm_pos = [0.00, 0.782, -1.4239, 3.2866, 2.459, -1.4822]

	contact_sensor_names = ["link_8.0_fsr", "link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
							"link_13.0_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
							"link_4.0_fsr", "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr",
							"link_0.0_fsr", "link_9.0_fsr", "link_10.0_fsr", "link_11.0_tip_fsr"]
	for bs in range(batch_size):
		for i, fsr in enumerate(contact_sensor_names):
			chain = pk.build_serial_chain_from_urdf(open(urdf_path, 'rb').read(), fsr)
			joint_num = len(chain.get_joint_parameter_names())
			# th = default_arm_pos + [0.0] * (joint_num - 6)
			if i < 4:
				th = default_arm_pos + joint_angles[bs,:joint_num-6].tolist()
			elif i >= 4 and i < 8:
				th = default_arm_pos + joint_angles[bs,4:4+joint_num-6].tolist()
			elif i >= 8 and i < 12:
				th = default_arm_pos + joint_angles[bs,8:8+joint_num-6].tolist()
			elif i >= 12 and i < 16:
				th = default_arm_pos + joint_angles[bs,12:12+joint_num-6].tolist()
			ret = chain.forward_kinematics(th, end_only=False)
			
			pos = ret[fsr].get_matrix()[:,:3,3].numpy()
			sensor_pos[bs,:,i] = torch.tensor(pos)
	return sensor_pos.view(batch_size,-1)



class NNRelativeMLPControllerMP:
	def __init__(self, dof_lower, dof_upper, num_actors=1, scale=0.5, stack=4,
				 config_path='./robot_controller/network.yaml'):

		self.num_actors = num_actors
		self.model = build_network(config_path, input_shape= 133 * stack)
		self.last_action = torch.zeros(num_actors, 22)
		self.target = torch.zeros(num_actors, 22)
		self.initial_target = torch.zeros(22)

		self.states = None
		self.dof_lower = torch.from_numpy(dof_lower)
		self.dof_upper = torch.from_numpy(dof_upper)
		self.all_commands = torch.tensor([[1.0, 0.0, 0.0],
										  [0.0, 1.0, 0.0],
										  [0.0, 0.0, 1.0]]).float()
		self.current_cmd = 0
		self.num_supported_cmd = 3

		self.need_init = 0
		self.last_action = torch.zeros(num_actors, 22)
		self.m_scale = scale

		self.single_step_obs_dim = 133
		self.stack = stack
		self.history = torch.zeros(num_actors, self.single_step_obs_dim * self.stack).float()

	# def reset_rnn(self):
	# 	rnn_states = self.model.get_default_rnn_state()
	# 	self.states = [torch.zeros((s.size()[0], self.num_actors, s.size(
	# 	)[2]), dtype=torch.float32) for s in rnn_states]

	def reset(self):
		self.target = self.initial_target.reshape(-1, 22).repeat(self.num_actors, 1)
		self.history = torch.zeros(self.num_actors, self.single_step_obs_dim * self.stack).float()
		self.need_init = 1
		# self.reset_rnn()

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
		obs = torch.from_numpy(observation).float()
		obs = self._preproc_obs(obs)
		# print(obs)
		expand_obs = torch.zeros((obs.size(0), 133))
		expand_obs[:,6:22] = obs[:,6:22]
		expand_obs[:, 22:45] = 0 #self.last_action
		expand_obs[:, 45:69] = self.get_command(obs.size(0))
		expand_obs[:, 69:85] = torch.zeros_like(obs[:, 45:61])
		expand_obs[:, 85:133] = get_sensor_pos(obs[:, 6:22])

		if self.need_init:
			self.history = expand_obs.repeat(1, self.stack)
			self.need_init = 0

		self.history = torch.cat((expand_obs, self.history[:, :-self.single_step_obs_dim]), dim=-1)
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

		self.last_action = current_action

		current_action = torch.clip(current_action, -1.0, 1.0)
		self.last_action = self.last_action * 0.2 + current_action * 0.8

		q_pos_delta = self.m_scale * self.last_action

		# Moving average.
		#target = self.scale(current_action)
		#self.target = 0.1 * target + 0.9 * self.target
		#self.target = torch.max(torch.min(self.target, self.dof_upper), self.dof_lower)

		return q_pos_delta.detach().cpu().numpy() #self.target.detach().cpu().numpy()

	def load(self, fn):
		checkpoint = torch_ext.load_checkpoint(fn)
		self.model.load_state_dict(checkpoint['model'])
		self.model.eval()

		# if self.normalize_input and 'running_mean_std' in checkpoint:
		#	self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


if __name__ == '__main__':
	DOF_LOWER_LIMITS = np.array([0.0, 0.782, -1.42391, 3.2866, 2.459, -1.48221, -0.47, -0.196, -0.174, -0.227,
								 0.263, -0.105, -0.189, -0.162, -0.47, -0.196, -0.174, -0.227, -0.47, -0.196,
								 -0.174, -0.227])

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
	model_name = 'baseline_mlp_encoder'
	model_path = os.path.join(model_root, model_name)

	network_config_path = os.path.join(model_path, 'network.yaml')
	network_checkpoint_path = get_model(model_path)
	policy = NNRelativeMLPControllerMP(dof_lower=DOF_LOWER_LIMITS, dof_upper=DOF_UPPER_LIMITS, num_actors=1,
						  		  	  config_path=network_config_path, scale=0.25)

	policy.load(network_checkpoint_path)
	policy.set_initial_target(INIT_QPOS)
	policy.reset()

	'''
		Dec 13:
		This one is new. 
		You can set command to 0(y-axis) or 1(z-axis) in the execution loop.
	'''
	policy.set_command(0)

	for i in range(100):
		obs = np.random.uniform(0, 1, 85)

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

		action = policy.predict(observation=obs.reshape(1, -1))[0]
		# TODO(): qpos = qpos + action
		# TODO(): set_qpos(qpos)
		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
			
			For relative control (This version), the update is: qpos_target = qpos + action * scale.
			For absolute control, the update is: qpos_target = rescale(action). The action is internally EMAed.	
		'''

		print(action)
