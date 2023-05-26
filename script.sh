CUDA_VISIBLE_DEVICES=3 python3 isaacgymenvs/train.py task=AllegroHandContinuous headless=True \
experiment=NewRotation1.0f0.04 task.env.spin_coef=1.0 task.env.fallDistance=0.04 \
&& CUDA_VISIBLE_DEVICES=3 python3 isaacgymenvs/train.py task=AllegroHandContinuous headless=True \
experiment=NewRotation1.0f0.06 task.env.spin_coef=1.0 task.env.fallDistance=0.06


python3 isaacgymenvs/train.py task=AllegroArmMOAR headless=True seed=-1 experiment=Baseline wandb_activate=True