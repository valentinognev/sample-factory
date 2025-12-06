from sample_factory.envs.trajectory_env import TrajectoryEnv, generate_circle_trajectory
import numpy as np

def create_trajectory_env(env_config):
    # You can pull parameters from env_config, e.g. radius, num_points, max_speed, dt, or even randomize per episode
    radius = env_config.get('radius', 3.0)
    num_points = env_config.get('num_points', 200)
    traj = generate_circle_trajectory(radius=radius, num_points=num_points)
    return TrajectoryEnv(
        traj,
        max_speed=env_config.get('max_speed', 0.5),
        dt=env_config.get('dt', 0.1)
    )

def create_2d_follower_env(env_config):
    # Example: could be same as trajectory but different defaults or logic
    radius = env_config.get('radius', 5.0)
    num_points = env_config.get('num_points', 300)
    traj = generate_circle_trajectory(radius=radius, num_points=num_points)
    return TrajectoryEnv(
        traj,
        max_speed=env_config.get('max_speed', 1.0),
        dt=env_config.get('dt', 0.05)
    )

ENV_REGISTRY['trajectory_point_follow'] = create_trajectory_env
ENV_REGISTRY['2d_follower'] = create_2d_follower_env