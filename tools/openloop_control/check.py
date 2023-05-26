import pickle
import sys
import time

import numpy as np

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_data(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
	# Example launch script:
	# python ./execute_control.py openloop.pkl 0
	openloop_trajectories_path = sys.argv[1]
	execute_id = 5

	openloop_trajectories = load_data(openloop_trajectories_path)

	execute_trajectory = openloop_trajectories[execute_id]['state'][:, :22]
	execute_action = openloop_trajectories[execute_id]['action'][:, :22]
	accumulation = []

	for t in range(1, 1000):
		errors = 0

		state = execute_trajectory[0]
		for i in range(len(execute_action)-1):
			
			state = state + execute_action[i] * 0.0001 * t
			error = execute_trajectory[i+1][6:] - state[6:]
			
			errors += np.abs(error).mean()
			#print("ERROR", error)
			# print((execute_trajectory[i+1] - execute_trajectory[i]) /  execute_action[i])
			accumulation.append((execute_trajectory[i+1] - execute_trajectory[i]) /  execute_action[i])
		
		print(t, errors)

	print("ACC",np.array(accumulation).mean(axis=0))
	# print("DS", execute_trajectory[3] - execute_trajectory[2])
	# print("S", execute_trajectory[2])
	# print("A", execute_action[2])

	# controller = Controller({})
	# controller.reset()
	# for action in execute_trajectory:
	# 	controller.send_relative_control(action * 20.0 * 0.1667)# action: target q_pos
