# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt
import time
from isaacgym import gymtorch
import pdb
import threading
import pytorch3d.transforms as transform

import matplotlib.pyplot as plt
import numpy
from scipy.spatial.transform import Rotation as R



xs = []
ys = []
ys_net = []
ys_inner = []

#
# def plot_thread():
#     hl, = plt.plot([], [])
#     plt.ion()
#     plt.show()
#     while True:
#         hl.set_xdata(xs)
#         hl.set_ydata(ys)
#         plt.draw()


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

'''
    Initialize the simulator.
'''

# configure sim
sim_params = gymapi.SimParams()

# set the up axis to be z-up given that assets are y-up by default
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81

if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(0, 0, args.physics_engine, sim_params)#args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
# contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

'''
    Create Global Ground Plane.
'''

# add ground plane
plane_params = gymapi.PlaneParams()
# set the normal force to be z dimension
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 0.01
plane_params.dynamic_friction = 0.01
plane_params.restitution = 1
gym.add_ground(sim, plane_params)


'''
    Create IsaacGym Viewer (GUI)
'''

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset") # subscribe to spacebar event for reset

'''
    Prepare Assets.
'''

# Load sensor asset.
asset_root = "./assets"
sensor_asset_file = "sensor_prism_rev.urdf"
sensor_asset = gym.load_asset(sim, asset_root, sensor_asset_file, gymapi.AssetOptions())
body_idx = gym.find_asset_rigid_body_index(sensor_asset, "sensor_link")
sensor_props = gymapi.ForceSensorProperties()
sensor_props.enable_forward_dynamics_forces = True
sensor_props.enable_constraint_solver_forces = True
sensor_props.use_world_frame = False

sensor_pose1 = gymapi.Transform(gymapi.Vec3(0.02, 0.0, 0.02))
sensor_idx1 = gym.create_asset_force_sensor(local_pose=sensor_pose1,
                                            body_idx=body_idx,
                                            asset=sensor_asset,
                                            props=sensor_props)



# Load box asset
box_asset_file = "box.urdf"
box_asset = gym.load_asset(sim, asset_root, box_asset_file, gymapi.AssetOptions())


# Setup multiple env instances in grid
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(0.5 * -env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(0.5 * env_spacing, env_spacing, env_spacing)
envs = []
all_sensors = []
all_actor_handles = []

# set random seed
np.random.seed(17)


'''
    Create the Environments.
'''

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # ---------------- create the sensor_actor -------------------------

    # Step 1: Setup initial pose.
    sensor_pose = gymapi.Transform()
    sensor_pose.p.z = 0.02
    sensor_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    collision_group = 0
    collision_filter = 0
    sensor_actor_handle = gym.create_actor(env, sensor_asset, sensor_pose, None, collision_group, collision_filter)
    all_actor_handles.append(sensor_actor_handle)
    gym.enable_actor_dof_force_sensors(env, sensor_actor_handle)

    box_pose = gymapi.Transform()
    box_pose.p.x = 0.6
    box_pose.p.z = 0.3
    box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    collision_group = 0
    collision_filter = 0
    box_actor_handle = gym.create_actor(env, box_asset, box_pose, None, collision_group, collision_filter)


    # box_pose = gymapi.Transform()
    # box_pose.p.x = 0.2
    # box_pose.p.z = 3.0
    # box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    # collision_group = 0
    # collision_filter = 0
    # box_actor_handle = gym.create_actor(env, box_asset, box_pose, None, collision_group, collision_filter)


    # Step 2: Setup up dof properties.
    # dof_props = gym.get_actor_dof_properties(env, sensor_actor_handle)
    # dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
    # # dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
    # dof_props['stiffness'][:] = 0.0
    # dof_props['damping'][:] = 0.0
    #gym.set_actor_dof_properties(env, sensor_actor_handle, dof_props)

    # Step 3: Setup physics properties
    shape_props = gym.get_actor_rigid_shape_properties(env, sensor_actor_handle)
    shape_props[0].friction = 0.
    shape_props[0].restitution = 1
    shape_props[0].compliance = 0.5
    # gym.set_actor_rigid_shape_properties(env, ahandle, shape_props)

    sensor = gym.get_actor_force_sensor(env, sensor_actor_handle, 0)
    all_sensors.append(sensor)


'''
    Main loop!
'''

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# import threading
# plotter = threading.Thread(target=plot_thread)
# plotter.start()

t = 0
plt.ion()
plt.show()
plt.ylim(-2, 2)
while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    sensor_net_force = all_sensors[0].get_forces().force
    #print("SENSOR FORCE", np.array([sensor_net_force.z, sensor_net_force.y, sensor_net_force.z]))
    sensor_net_force = np.array([sensor_net_force.z, sensor_net_force.y, sensor_net_force.z])

    forces = gym.get_actor_dof_forces(envs[0], all_actor_handles[0])
    #print("SENSOR BAR FORCE", forces)


    # We also need to get the rotation direction.
    body_states = gym.get_actor_rigid_body_states(envs[0], all_actor_handles[0], gymapi.STATE_ALL)
    #print(body_states)
    quat = body_states[0][0][1]
    #print(quat)

    # Convert it to rotation matrix. Get the direction vector.
    r = R.from_quat([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])
    main_direction = np.array(r.apply([0., 0., 1.]))
    #print(main_direction)

    gravity = np.array([0.0, 0.0, -9.8])
    mass = 1.0 * 0.5 * 0.5 * 0.02
    gravity_force = mass * gravity

    #print("NF", sensor_net_force, "F0")
    # contact_force = np.inner(sensor_net_force, main_direction) \
    #                 - np.inner(gravity_force, main_direction) \
    #                 - forces[0]

    _net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)
    print(net_cf[:3])
    contact_force = np.linalg.norm(net_cf[:3][2].detach().cpu().numpy())

    #print(contact_force)

    # Using our equation to get the contact force.

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    time.sleep(0.01)

    xs.append(t)
    ys.append(contact_force)
    #ys_net.append(np.inner(sensor_net_force, main_direction))
    #ys_inner.append(forces[0])
    plt.clf()
    #plt.ylim(-10, 10)
    plt.plot(xs, ys, 'b')
    # plt.plot(xs, ys_net, 'r')
    # plt.plot(xs, ys_inner, 'g')
    plt.plot()
    plt.draw()
    plt.pause(0.001)

    t += 0.02

plotter.join()
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
