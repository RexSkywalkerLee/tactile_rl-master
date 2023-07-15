CUDA_VISIBLE_DEVICES=3 python3 isaacgymenvs/train.py task=AllegroHandContinuous headless=True \
experiment=NewRotation1.0f0.04 task.env.spin_coef=1.0 task.env.fallDistance=0.04 \
&& CUDA_VISIBLE_DEVICES=3 python3 isaacgymenvs/train.py task=AllegroHandContinuous headless=True \
experiment=NewRotation1.0f0.06 task.env.spin_coef=1.0 task.env.fallDistance=0.06


python3 isaacgymenvs/train.py task=AllegroArmMOAR headless=True seed=42 experiment=Baseline_sd42 wandb_activate=False \
train.params.network.use_pretrain_tactile=True train.params.config.central_value_config.network.use_pretrain_tactile=True

python3 isaacgymenvs/train.py task=AllegroArmMOAR headless=True seed=42 experiment=RgBaselineC4_1 wandb_activate=True train.params.network.use_pretrain_tactile=True train.params.config.central_value_config.network.use_pretrain_tactile=True

python3 isaacgymenvs/train.py test=True task=AllegroArmMOAR headless=False seed=42 experiment=test wandb_activate=False train.params.network.use_pretrain_tactile=False train.params.config.central_value_config.network.use_pretrain_tactile=False task.env.numEnvs=27 checkpoint=baseline.pth