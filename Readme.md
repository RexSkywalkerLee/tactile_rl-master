# In-Hand Reorientation with Contact 

### Preparation
Install the latest **isaacgym** on your laptop (should have a GPU) and server (follow the instructions on the NVIDIA's website, you need to register an account).

### Launch Training
First, check the training environment on the laptop with
```
python ./isaacgymenvs/train.py task=AllegroArmContinuous headless=False
```

Then, launch training with

```
python3 isaacgymenvs/train.py task=AllegroArmContinuous headless=True experiment=ArmRotation1.0f0.10y task.env.spin_coef=1.0 task.env.axis=y task.env.observationType=partial_contact task.env.fallDistance=0.10 task.env.numEnvs=16384 train.params.config.minibatch_size=32768
```

In the script, ```task.env.axis=y``` indicates that we would like the hand to rotate the cube continuous around the y-axis (world frame). ```task.env.observationType=partial_contact``` indicates that we only use the sensor contact information. We can also use ```task.env.observationType=full_contact``` to use all the contact information in the simulation.

Besides, we can also use ```task.env.objectType=egg``` to use egg instead of cube for training. We will add more objects for training.
### Inspect the model
Finally, you can pull the model ```MODEL_NAME.pth``` from the ```./runs/EXPERIMENT_NAME/nn``` folder, and inspect it on your laptop.

To do this, run this on your laptop:

```
python ./isaacgymenvs/train.py task=AllegroArmContinuous test=True checkpoint=./MODEL_NAME.pth num_envs=8
```

### Ongoing stuffs.

Automate the workflow, including video sync over wandb. 

Improve the reward design.