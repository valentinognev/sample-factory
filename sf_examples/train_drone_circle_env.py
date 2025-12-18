"""
From the root of Sample Factory repo this can be run as:
python -m sf_examples.train_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example_multi --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.enjoy_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example_multi

"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

# Matplotlib import is done inside render() to avoid requiring it for training

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.train import run_rl
from sf_examples.train_custom_env_custom_model import make_custom_encoder, override_default_params


class DroneCircleEnv(gym.Env, TrainingInfoInterface, RewardShapingInterface):
    """
    Multi-agent environment where drones learn to form a circle at a target radius
    while maintaining spacing between neighbors.
    """

    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)

        self.name = full_env_name
        self.cfg = cfg

        self.num_agents = 8  # N drones
        self.is_multiagent = True
        self.target_radius = 5.0

        # Action: [Velocity_X, Velocity_Y]
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Observation: [My_X, My_Y, Closest_Neighbor_RX, Closest_Neighbor_RY]
        # We normalize inputs roughly to [-1, 1] range for better learning
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

        self.agents_pos = np.zeros((self.num_agents, 2))
        self.episode_len = getattr(cfg, "custom_env_episode_len", 200)
        self.curr_step = 0
        self.reward_shaping = [dict(rew=1.0) for _ in range(self.num_agents)]

        self.render_mode = render_mode
        self._render_fig = None
        self._render_ax = None

    def reset(self, **kwargs):
        self.curr_step = 0
        # Spawn agents randomly near the center
        self.agents_pos = np.random.uniform(-10, 10, (self.num_agents, 2))
        return self._get_obs(), [dict() for _ in range(self.num_agents)]

    def _get_obs(self):
        obs_list = []
        for i in range(self.num_agents):
            me = self.agents_pos[i]

            # Find closest neighbor
            # Calculate all distances
            diffs = self.agents_pos - me  # (N, 2)
            dists = np.linalg.norm(diffs, axis=1)  # (N,)

            # Mask myself (distance 0) with infinity so I'm not my own neighbor
            dists[i] = np.inf

            # Handle edge case where all agents might be at same position
            if np.all(np.isinf(dists)):
                # If all distances are inf (shouldn't happen with >1 agent), use zero vector
                closest_vec = np.zeros(2)
            else:
                closest_idx = np.argmin(dists)
                closest_vec = diffs[closest_idx]  # Vector pointing TO neighbor

            # Observation: My Pos + Vector to Closest Neighbor
            # We scale down by target_radius to keep inputs near 0-1
            # Clip observations to prevent extreme values
            obs_vec = np.concatenate([me, closest_vec]) / self.target_radius
            obs_vec = np.clip(obs_vec, -10.0, 10.0)  # Prevent extreme observations
            obs_list.append(np.float32(obs_vec))

        return obs_list

    def step(self, actions):
        # 1. Apply Actions (Velocity Control)
        # Scale action to max speed (e.g., 0.5 units per step)
        # Handle actions which may come as list of arrays or already as array
        if isinstance(actions, list):
            actions = np.array([np.asarray(a) for a in actions])
        else:
            actions = np.asarray(actions)
        
        # Ensure actions have correct shape: (num_agents, 2)
        if actions.shape != (self.num_agents, 2):
            if actions.ndim == 1:
                # Single action for all agents (unlikely but handle it)
                actions = np.tile(actions, (self.num_agents, 1))
            else:
                actions = actions.reshape(self.num_agents, 2)
        
        self.agents_pos += actions * 0.5
        
        # Clip positions to prevent agents from moving too far away (optional, but helps stability)
        # self.agents_pos = np.clip(self.agents_pos, -20.0, 20.0)

        rewards = []
        infos = [dict(is_active=True) for _ in range(self.num_agents)]

        # 2. Calculate Rewards
        for i in range(self.num_agents):
            me = self.agents_pos[i]

            # A. Circle Reward: Minimize distance to Target Radius
            dist_to_center = np.linalg.norm(me)
            # Penalty is squared error from radius
            r_circle = -np.abs(dist_to_center - self.target_radius)

            # B. Spreading Reward: Maximize distance to nearest neighbor
            # (Re-calculate simply for clarity)
            dists = np.linalg.norm(self.agents_pos - me, axis=1)
            dists[i] = np.inf
            dist_closest = np.min(dists)

            # We want to encourage distance, but cap it (diminishing returns)
            # If we don't cap it, they might leave the circle just to get away
            # r_spread = np.clip(dist_closest, 0, 3.0)
            r_spread = dist_closest

            # Total Reward
            # We weight keeping the circle shape higher than spreading
            total_rew = (r_circle * 1.0) + (r_spread * 0.5)
            
            # Safety check: prevent NaN or inf rewards
            if not np.isfinite(total_rew):
                total_rew = 0.0
            
            rewards.append(float(total_rew))

        self.curr_step += 1
        terminated = truncated = [self.curr_step >= self.episode_len] * self.num_agents

        if terminated[0]:
            # multi-agent environments should auto-reset!
            obs, infos = self.reset()
        else:
            obs = self._get_obs()

        return obs, rewards, terminated, truncated, infos

    def get_default_reward_shaping(self) -> Optional[Dict[str, Any]]:
        return self.reward_shaping[0]

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        if isinstance(agent_idx, int):
            agent_idx = slice(agent_idx, agent_idx + 1)
        for idx in range(agent_idx.start, agent_idx.stop):
            self.reward_shaping[idx] = reward_shaping

    def render(self):
        """Render the environment using matplotlib."""
        import matplotlib.pyplot as plt
        
        if self.render_mode == "human" or self.render_mode is None:
            if self._render_fig is None:
                plt.ion()  # Turn on interactive mode
                self._render_fig, self._render_ax = plt.subplots(figsize=(10, 10))
                self._render_ax.set_aspect('equal')
                self._render_ax.set_xlim(-15, 15)
                self._render_ax.set_ylim(-15, 15)
                self._render_ax.grid(True, alpha=0.3)
                self._render_ax.set_title("Drone Circle Environment")
                self._render_ax.set_xlabel("X Position")
                self._render_ax.set_ylabel("Y Position")
                plt.tight_layout()
            
            self._render_ax.clear()
            self._render_ax.set_aspect('equal')
            self._render_ax.set_xlim(-15, 15)
            self._render_ax.set_ylim(-15, 15)
            self._render_ax.grid(True, alpha=0.3)
            self._render_ax.set_title(f"Drone Circle Environment - Step {self.curr_step}/{self.episode_len}")
            self._render_ax.set_xlabel("X Position")
            self._render_ax.set_ylabel("Y Position")
            
            # Draw target circle
            circle = plt.Circle((0, 0), self.target_radius, fill=False, color='green', 
                              linestyle='--', linewidth=2, label='Target Circle')
            self._render_ax.add_patch(circle)
            
            # Draw drones
            for i, pos in enumerate(self.agents_pos):
                self._render_ax.scatter(pos[0], pos[1], s=100, alpha=0.7, 
                                      label=f'Drone {i}' if i < 3 else '')
            
            # Draw lines to closest neighbors
            for i in range(self.num_agents):
                me = self.agents_pos[i]
                diffs = self.agents_pos - me
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = np.inf
                if not np.all(np.isinf(dists)):
                    closest_idx = np.argmin(dists)
                    closest_pos = self.agents_pos[closest_idx]
                    self._render_ax.plot([me[0], closest_pos[0]], [me[1], closest_pos[1]], 
                                       'r--', alpha=0.3, linewidth=1)
            
            # Add legend (only show first few to avoid clutter)
            if self.num_agents <= 8:
                self._render_ax.legend(loc='upper right', fontsize=8)
            else:
                self._render_ax.text(0.02, 0.98, f'{self.num_agents} drones', 
                                   transform=self._render_ax.transAxes, 
                                   verticalalignment='top', fontsize=10)
            
            plt.draw()
            plt.pause(0.01)  # Small pause to allow the plot to update
            
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Drone Circle Environment - Step {self.curr_step}/{self.episode_len}")
            
            # Draw target circle
            circle = plt.Circle((0, 0), self.target_radius, fill=False, color='green', 
                              linestyle='--', linewidth=2)
            ax.add_patch(circle)
            
            # Draw drones
            for i, pos in enumerate(self.agents_pos):
                ax.scatter(pos[0], pos[1], s=100, alpha=0.7)
            
            # Draw lines to closest neighbors
            for i in range(self.num_agents):
                me = self.agents_pos[i]
                diffs = self.agents_pos - me
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = np.inf
                if not np.all(np.isinf(dists)):
                    closest_idx = np.argmin(dists)
                    closest_pos = self.agents_pos[closest_idx]
                    ax.plot([me[0], closest_pos[0]], [me[1], closest_pos[1]], 
                           'r--', alpha=0.3, linewidth=1)
            
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return buf
        
        return None


def make_drone_circle_env_func(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return DroneCircleEnv(full_env_name, cfg, render_mode=render_mode)


def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_episode_len", default=200, type=int, help="Number of steps in the episode")


def register_custom_components():
    register_env("drone_circle_env", make_drone_circle_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
