from rl_games.algos_torch.model_builder import ModelBuilder
import yaml
import os

def read_yaml(yaml_path):
	with open(yaml_path, "r") as stream:
		try:
			result = yaml.safe_load(stream)
			print(result)
		except yaml.YAMLError as exc:
			print(exc)
	return result

def get_model(checkpoint_path):
	all_files = os.listdir(checkpoint_path)
	for f in all_files:
		if '.pth' in f:
			return os.path.join(checkpoint_path, f)
	assert False, "no model found in " + checkpoint_path
	return ''

def build_network(cfg_yaml_path, input_shape=60, normalize=False):
	cfg = read_yaml(cfg_yaml_path)
	model_builder = ModelBuilder()
	model = model_builder.load(cfg['params'])

	config = {
		'actions_num': 22,
		'input_shape': (input_shape, ),
		'num_seqs': 1,
		'value_size': 1,
		'normalize_value': True, #self.normalize_value,
		'normalize_input': normalize #,self.normalize_input,
	}

	network = model.build(config)
	print(network)
	return network

if __name__ == '__main__':
	#read_yaml('./robot_controller/network.yaml')
	get_model('./robot_controller/models/x_lstm_dec27_slow')

