
script_prefix = 'crotate-'

bash_script_magical = 'CUDA_VISIBLE_DEVICES=$1 python3 ./isaacgymenvs/train.py ' \
                      'headless=True ' \
                      'train.params.config.user_prefix=ENTB ' \
                      'task=AllegroArmLeftContinuous ' \
                      'task.env.rewardType=free ' \
                      'task.task.randomize=True ' \
                      'task.env.sensor=normal ' \
                      'task.env.main_coef={main} ' \
                      'task.env.contact_coef={cont} ' \
                      'task.env.spin_coef={spin} ' \
                      'task.env.handInit=default ' \
                      'task.env.axis={axis} ' \
                      'task.env.numEnvs=16384 ' \
                      'train.params.config.minibatch_size=32768 ' \
                      'train.params.config.entropy_coef={ent}'


axis = ['z']
# ent = [0.00]
ent = [0.01, 0.001, 0.003, 0.0003, 0.0001, 0.03]
main = [0.00]
cont = [0.00]
spin = [1.0]

def generate_gridsearch_from_lists(names, lists):
    if len(lists) == 1:
        return [{names[0]: i} for i in lists[0]]
    elif len(lists) > 1:
        prev_results = generate_gridsearch_from_lists(names[1:], lists[1:])
        new_results = []

        for result in prev_results:
            for i in lists[0]:
                new_result = result.copy()
                new_result[names[0]] = i
                new_results.append(new_result)
        return new_results
    else:
        return []


def generate_series(template, arg_name, arg_val):
    gridsearch_dicts = generate_gridsearch_from_lists(arg_name, arg_val)
    all_launch_script = [template.format(**d) for d in gridsearch_dicts]
    return all_launch_script


#all_launch_script = generate_series(bash_script_procgen, ['env_name', 'da', 'seed'], [env_name, da, seed])
all_launch_script = generate_series(bash_script_magical, ['axis', 'ent', 'cont', 'spin', 'main'], [axis, ent, cont, spin, main])

num_processors = 3
for i in range(num_processors):
    length = len(all_launch_script) // num_processors
    if i < num_processors - 1:
        scripts = all_launch_script[length * i : length * (i+1)]
    else:
        scripts = all_launch_script[length * i : ]

    file_name = '../launch/{}.sh'.format(script_prefix + str(i))

    with open(file_name, 'w+') as f:
        for script in scripts:
            f.write(script + '\n')

