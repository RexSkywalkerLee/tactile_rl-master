import json

def read_dict_from_json(file_path):
	# Opening JSON file
	with open(file_path) as json_file:
		data = json.load(json_file)
	return data


if __name__ == '__main__':
	print(read_dict_from_json('../isaacgymenvs/misc/egad_asset.txt').keys())