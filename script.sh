CUDA_VISIBLE_DEVICES=3 python3 isaacgymenvs/train.py task=AllegroHandContinuous headless=True \
experiment=NewRotation1.0f0.04 task.env.spin_coef=1.0 task.env.fallDistance=0.04 \
&& CUDA_VISIBLE_DEVICES=3 python3 isaacgymenvs/train.py task=AllegroHandContinuous headless=True \
experiment=NewRotation1.0f0.06 task.env.spin_coef=1.0 task.env.fallDistance=0.06


python3 isaacgymenvs/train.py task=AllegroArmMOAR headless=True seed=37 experiment=Pretrained_sd37 wandb_activate=True