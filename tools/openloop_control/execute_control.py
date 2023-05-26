import pickle
import sys
import time

import numpy as np

# lower limits of each joint
DOF_LOWER_LIMITS = np.array([0.0, 0.782, -1.087, 3.187, 2.109,
							-1.615, -0.47, -0.196, -0.174, -0.227,
							0.263, -0.105, -0.189, -0.162, -0.47,
							-0.196, -0.174, -0.227, -0.47, -0.196,
							-0.174, -0.227])

# upper limits of each joint
DOF_UPPER_LIMITS = np.array([0.001, 0.783, -1.086, 3.188, 2.11,
							-1.614, 0.47, 1.61, 1.709, 1.618,
							1.396, 1.163, 1.644, 1.719, 0.47,
							1.61, 1.709, 1.618, 0.47, 1.61,
							1.709, 1.618])

DEFAULT_HAND_QPOS = {
	"joint_0.0": 0.0261,
	"joint_1.0": 0.5032,
	"joint_2.0": 0.0722,
	"joint_3.0": 0.7050,
	"joint_12.0": 0.8353,
	"joint_13.0": -0.0388,
	"joint_14.0": 0.3703,
	"joint_15.0": 0.3444,
	"joint_4.0": 0.0048,
	"joint_5.0": 0.6514,
	"joint_6.0": -0.0147,
	"joint_7.0": 0.4276,
	"joint_8.0": -0.0868,
	"joint_9.0": 0.4106,
	"joint_10.0": 0.3233,
	"joint_11.0": 0.2792
}

DEFAULT_ARM_QPOS = {
	'joint1': 0.00, 'joint2': 0.782, 'joint3': -1.087,
	'joint4': 3.187, 'joint5': 2.109, 'joint6': -1.615
}

# The definition of each action vector.
DOF_ORDER = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5',
			 'joint6', 'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
			 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0',
			 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_8.0',
			 'joint_9.0', 'joint_10.0', 'joint_11.0']


def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_data(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


# To Binghao:
# Openloop testing routine.
def setup():
	# TODO()
	pass


class Controller:
	def __init__(self, ros_config):
		# TODO()
		self.ros_config = ros_config
		pass

	def reset(self):
		self.ros_send(DEFAULT_HAND_QPOS)
		self.ros_send(DEFAULT_ARM_QPOS)
		self.prev_target = self.description_to_numpy(DEFAULT_HAND_QPOS)
		self.curr_target = self.prev_target
		time.sleep(3.0)

	def description_to_numpy(self, data):
		# TODO()
		return data

	def numpy_to_description(self, data):
		# TODO()
		return data

	def ros_send(self, x):
		# TODO()
		return
	def clamp(self, x, lower, upper):
		# TODO()
		return x
	def send_relative_control(self, action):
		self.curr_target = self.prev_target + action
		self.curr_target = self.clamp(self.curr_target, DOF_LOWER_LIMITS, DOF_UPPER_LIMITS)
		self.prev_target = self.curr_target
		self.ros_send(self.numpy_to_description(self.curr_target))


if __name__ == '__main__':
	# Example launch script:
	# python ./execute_control.py openloop.pkl 0
	openloop_trajectories_path = sys.argv[1]
	execute_id = int(sys.argv[2])

	openloop_trajectories = load_data(openloop_trajectories_path)

	execute_trajectory = openloop_trajectories[execute_id]['state'][:, :22]
	print(execute_trajectory.shape)

	controller = Controller({})
	controller.reset()
	for action in execute_trajectory:
		controller.send_relative_control(action * 20.0 * 0.1667)# action: target q_pos
