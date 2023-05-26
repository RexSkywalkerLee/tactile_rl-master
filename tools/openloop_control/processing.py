import pickle
import sys
import numpy as np

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_data(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
	trajectory_source = sys.argv[1]
	trajectory_target = sys.argv[2]

	trajectories = load_data(trajectory_source)

	num_steps, num_envs = trajectories['act'].shape[0], trajectories['act'].shape[1]
	print(num_steps, num_envs)
	all_trajs = []
	tmp_traj_actions = []
	tmp_traj_states = []

	trajectory_states = trajectories['obs']
	trajectory_actions = trajectories['act']
	trajectory_dones = trajectories['done']

	for env in range(num_envs):
		for step in range(num_steps):
			tmp_traj_actions.append(trajectory_actions[step, env])
			tmp_traj_states.append(trajectory_states[step, env])

			if trajectory_dones[step, env]:
				print(np.array(tmp_traj_states).shape)
				if len(tmp_traj_actions) > 100:
					all_trajs.append({'state': np.array(tmp_traj_states),
									  'action': np.array(tmp_traj_actions)})

				tmp_traj_states = []
				tmp_traj_actions = []


		if len(tmp_traj_states) > 0:
			print(np.array(tmp_traj_states).shape)
			if len(tmp_traj_actions) > 150:
				all_trajs.append({'state': np.array(tmp_traj_states),
								  'action': np.array(tmp_traj_actions)})

			tmp_traj_states = []
			tmp_traj_actions = []

	save_data(all_trajs, trajectory_target)
