import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sample_factory.envs.env_utils import TrainingInfoInterface

import csv, os, time, atexit
from datetime import datetime

class PointTrajectoryEnv(gym.Env, TrainingInfoInterface):
    """
    Gymnasium environment for a 2D drone tracking a moving target.
    Actions: polar velocities [v_r, v_theta, w_yaw].
    Observations: [r, theta, radial_rate].
    Rewards: weighted MSE on r and theta, LOS bonus, radial closure bonus.
    Target follows either a preset trajectory (circle/square) or OU random motion.
    Visualization: top-down and first-person (human) via matplotlib.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, full_env_name: str, cfg=None, env_config=None, render_mode=None):
        super().__init__()
        self.dt = 0.1
        self.render_mode = render_mode

        # Base parameters
        self.radius_mean = 10.0
        self.radius_min = 0.0
        self.radius_max = 20.0
        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.vf = 0.0
        self.vr = 0.0
        self.w = 0.0
        self.tau_v = 0.8 # time to reach 66% of v commands in sec
        self.tau_w = 0.1 # time to reach 66% of w command in sec

        # OU random parameters
        self.ou_theta_r = 0.7; self.ou_sigma_r = 0.3; self.ou_r_dev = 0.0
        self.ou_theta_a = 0.5; self.ou_sigma_a = 0.5; self.ou_a_drift = 0.3

        # Reward shaping
        self.r_reward = 0.4
        self.theta_reward = 33.0
        self.reward_min = 0.5
        self.reward_max = 2.5
        self.goal_factor = 10
        self.goal_radius = 0.2
        # Curriculum: action-penalty parameters
        self.init_AP = 0.0
        self.max_AP = 5.0
        self.schedule_steps = 1e7
        self.AP_factor = self.init_AP  # start with minimal penalty

        # Episode length
        self.max_steps = getattr(env_config, 'max_steps', 1000)

        # Field of view
        self.fov = 2 * np.pi / 3
        self.fp_dist_max = 20.0

        # Action & observation spaces
        # action space = [forward velocity, right velocity, yaw speed (currently offline)]
        # observation space = [r , theta, r_dot, theta_dot, vf, vr, vf_avg, vr_avg]
        self.max_speed = 2.0
        self.max_w = np.deg2rad(30)
        self.action_space = spaces.Box(
            low=np.array([-self.max_speed, -self.max_speed, -self.max_w ], dtype=np.float32),
            high=np.array([ self.max_speed,  self.max_speed,  self.max_w ], dtype=np.float32),
            dtype=np.float32
        )
        obs_low  = np.array([0.0, -np.pi, -self.max_speed, -np.pi, 
                             -self.max_speed, -self.max_speed, -self.max_speed, -self.max_speed], dtype=np.float32)
        obs_high = np.array([20.0,  np.pi, self.max_speed, np.pi, 
                             self.max_speed, self.max_speed, self.max_speed, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_heading = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_angle = 0.0
        self.current_radius = 0.0
        self.step_count = 0
        # Action history
        self.history_length = 10
        self.vf_history = np.zeros(self.history_length)
        self.vr_history = np.zeros(self.history_length)
        self.vf_avg = 0.0
        self.vr_avg = 0.0
        self.hist_i = 0 

        # Debug parameters
        self.raw_vf = 0.0
        self.raw_vr = 0.0
        self.raw_w = 0.0
        # Episode reward component accumulators for TensorBoard
        self.ep_stats = dict(dist=0.0, angle=0.0, act_pen=0.0, total=0.0)
        self.last_r = None
        self.last_theta = None

        # Trajectory type: 'circle', 'square', or 'random'
        self.trajectory_type = getattr(cfg, 'trajectory', 'random')
        if self.trajectory_type != 'random':
            self._generate_trajectory(self.trajectory_type)
            self.current_index = 0
        self.waypoint = (-10,0)

        # -- CSV LOGGING SETUP: only raw actions ----------------------
        log_dir = os.path.join('logs', getattr(cfg, 'experiment', ''))
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(log_dir, f'data_log_{ts}.csv')
        self._log_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._log_file)
        self._csv_writer.writerow(["time", "raw_vf", "raw_vr", "raw_w"])
        self._log_file.flush()
        atexit.register(self._log_file.close)
        # --------------------------------------------------------------

        self.theta = 0
        self.vf_cmd = 0
        self.vr_cmd = 0
        self.w_cmd = 0

    def set_training_info(self, training_info: dict):
        """
        Callback from Sample Factory with training progress.
        Adjusts self.AP_factor on a linear schedule from init_AP to max_AP.
        """
        super().set_training_info(training_info)
        steps = training_info.get('approx_total_training_steps', 0)
        frac = min(1.0, steps / self.schedule_steps)
        self.AP_factor = self.init_AP + frac * (self.max_AP - self.init_AP)

    def _generate_trajectory(self, traj_type: str):
        N = 75
        if traj_type == 'circle':
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            self.trajectory_points = np.stack([
                self.radius_mean * np.cos(angles),
                self.radius_mean * np.sin(angles)
            ], axis=1)
        elif traj_type == 'square':
            per = N // 4
            pts = []
            # bottom edge
            for i in range(per): pts.append((-self.radius_mean + 2*self.radius_mean*i/per, -self.radius_mean))
            # right edge
            for i in range(per): pts.append(( self.radius_mean, -self.radius_mean + 2*self.radius_mean*i/per))
            # top edge
            for i in range(per): pts.append(( self.radius_mean - 2*self.radius_mean*i/per,  self.radius_mean))
            # left edge
            for i in range(per): pts.append((-self.radius_mean,  self.radius_mean - 2*self.radius_mean*i/per))
            self.trajectory_points = np.array(pts, dtype=np.float32)
        elif traj_type == 'waypoint' or traj_type == 'waypoint2':
            waypoint = (-10,0)
            self.trajectory_points = np.full((N, 2), waypoint, dtype=np.float32)
        else:
            raise ValueError(f"Unknown trajectory: {traj_type}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset agent
        self.agent_pos[:] = 0.0
        self.agent_heading = 0.0
        self.vf_history = np.zeros(self.history_length)
        self.vr_history = np.zeros(self.history_length)
        self.vf_avg = 0.0
        self.vr_avg = 0.0
        self.hist_i = 0 

        # Reset OU
        self.ou_r_dev = 0.0
        self.ou_a_drift = 0.0

        # Reset per-episode accumulators for TB
        self.ep_stats = dict(dist=0.0, angle=0.0, act_pen=0.0, total=0.0)

        # Initialize target
        if self.trajectory_type == 'waypoint':
            x = np.random.uniform(-15.0,15.0)
            y = np.random.uniform(-15.0,15.0)
            self.waypoint = (x,y)
            self.agent_heading = np.random.uniform(-np.pi,np.pi)
        elif self.trajectory_type == 'waypoint2':
            x = np.random.uniform(-1.0,1.0)
            y = np.random.uniform(-1.0,1.0)
            self.waypoint = (x,y)
        elif self.trajectory_type != 'random':
            self.current_index = 0
            pt = self.trajectory_points[0]
            self.target_pos    = pt.copy()
            self.current_radius= np.linalg.norm(pt)
            self.target_angle  = np.arctan2(pt[1], pt[0])
        else:
            self.target_angle = self.np_random.uniform(self.angle_min, self.angle_max)
            self.current_radius = self.radius_mean
            self.target_pos = np.array([
                self.current_radius * np.cos(self.target_angle),
                self.current_radius * np.sin(self.target_angle)
            ], dtype=np.float32)

        # Initial obs
        r, theta = self._get_obs()
        self.theta = theta ## DEBUG
        self.last_r = r
        self.last_theta = theta
        obs = np.array([r, theta,0.0, 0.0, 0.0, 0.0, self.vf_avg, self.vr_avg], dtype=np.float32)
        self.step_count = 0
        if self.render_mode == 'human':
            self._init_render()
        return obs, {}

    def _get_obs(self):
        delta = self.target_pos - self.agent_pos
        r = np.linalg.norm(delta)
        angle_to_target = np.arctan2(delta[0], delta[1])
        theta = (angle_to_target - self.agent_heading + np.pi) % (2*np.pi) - np.pi
        return r, theta

    def step(self, action):
        # populate raw actions for DEBUG
        self.raw_vf, self.raw_vr, self.raw_w = action
        # Agent move
        v_f_cmd, v_r_cmd, w_cmd = np.clip(action, self.action_space.low, self.action_space.high)

        # 2) First-order dynamics update
        alpha = np.exp(-self.dt/self.tau_v)
        self.vf = alpha * self.vf + (1.0 - alpha) * v_f_cmd
        self.vr = alpha * self.vr + (1.0 - alpha) * v_r_cmd

        beta = np.exp(-self.dt/self.tau_w)
        self.w  = 0.0 #beta * self.w + (1.0 - beta) * w_cmd

        dN = self.vf*np.cos(self.agent_heading) - self.vr*np.sin(self.agent_heading)
        dE = self.vf*np.sin(self.agent_heading) + self.vr*np.cos(self.agent_heading)
        self.agent_pos += np.array([dE, dN], dtype=np.float32)*self.dt
        self.agent_heading += self.w*self.dt

        # Target move
        if self.trajectory_type != 'random':
            self.current_index = (self.current_index + 1) % len(self.trajectory_points)
            pt = self.trajectory_points[self.current_index]
            self.target_pos     = pt.copy()
            self.current_radius = np.linalg.norm(pt)
            self.target_angle   = np.arctan2(pt[0], pt[1])
        else:
            # OU radius
            dr = self.ou_theta_r*(0.0 - self.ou_r_dev)*self.dt + self.ou_sigma_r*np.sqrt(self.dt)*self.np_random.standard_normal()
            self.ou_r_dev += dr
            self.current_radius = np.clip(self.radius_mean + self.ou_r_dev, self.radius_min, self.radius_max)
            # OU angle
            da = self.ou_theta_a*(0.0 - self.ou_a_drift)*self.dt + self.ou_sigma_a*np.sqrt(self.dt)*self.np_random.standard_normal()
            self.ou_a_drift += da
            self.target_angle += self.ou_a_drift*self.dt
            self.target_angle = ((self.target_angle - self.angle_min) % (self.angle_max - self.angle_min)) + self.angle_min
            self.target_pos = np.array([
                self.current_radius*np.cos(self.target_angle),
                self.current_radius*np.sin(self.target_angle)
            ], dtype=np.float32)

        # Observation
        r, theta = self._get_obs()
        self.theta = theta ## DEBUG
        r_dot = (r - self.last_r)/self.dt
        theta_dot = (theta - self.last_theta)/self.dt
        self.last_r = r
        self.last_theta = theta
            # Save action history and compute running average
        self.vf_history[self.hist_i] = self.vf
        self.vr_history[self.hist_i] = self.vr
        self.vf_avg = np.sum(self.vf_history)/self.history_length
        self.vr_avg = np.sum(self.vr_history)/self.history_length
        self.hist_i = (self.hist_i + 1) % self.history_length
        obs = np.array([r, theta, r_dot, theta_dot, 
                        self.vf, self.vr, self.vf_avg, self.vr_avg], dtype=np.float32)

        # Reward
        dist_part = -self.r_reward*(r**2)
        angle_part = -(self.theta_reward * (np.sqrt(np.abs(theta))) *
                       np.clip((np.abs(r) - self.reward_min) / (self.reward_max - self.reward_min), 0.0, 1.0))
        action_penalty = -self.AP_factor*(np.abs(self.vf) + np.abs(self.vr))
        avg_action_penalty = -self.AP_factor*(np.abs(self.vf-self.vf_avg) + np.abs(self.vr-self.vr_avg))
        goal_reward = self.goal_factor if r<self.goal_radius else 0.0
        reward = dist_part + 0.0*angle_part + action_penalty + avg_action_penalty + goal_reward

        # Accumulate per-episode reward components for TensorBoard
        self.ep_stats['dist']    += float(dist_part)
        self.ep_stats['angle']   += float(angle_part)
        self.ep_stats['act_pen'] += float(action_penalty)
        self.ep_stats['total']   += float(reward)

        self.vf_cmd = v_f_cmd
        self.vr_cmd = v_r_cmd
        self.w_cmd = w_cmd

        # --- CSV logging (only raw actions) ------------------------
        self._csv_writer.writerow([time.time(), self.raw_vf, self.raw_vr, self.raw_w])
        self._log_file.flush()
        # -----------------------------------------------------------

        # Termination
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        info = {"r":float(r), "theta":float(theta), "r_dot":float(r_dot), "theta_dot":float(theta_dot)}
        # Attach episode-end stats so Sample Factory logs to TensorBoard
        if truncated:
            steps = max(1, self.step_count)
            info['episode_extra_stats'] = {
                'R_ep/dist_sum':  float(self.ep_stats['dist']),
                'R_ep/angle_sum': float(self.ep_stats['angle']),
                'R_ep/action_sum': float(self.ep_stats['act_pen']),
                'R_ep/total_sum': float(self.ep_stats['total']),
                'R_ep/dist_mean':  float(self.ep_stats['dist'])  / steps,
                'R_ep/angle_mean': float(self.ep_stats['angle']) / steps,
                'R_ep/act_mean':   float(self.ep_stats['act_pen'])/ steps,
                'R_ep/total_mean': float(self.ep_stats['total']) / steps,
            }
        if self.render_mode=='human': self.render()
        return obs, reward, False, truncated, info

    def _init_render(self):
        import matplotlib.pyplot as plt
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number): return
        self.fig, (self.ax_top, _) = plt.subplots(1,2,figsize=(10,5))
        self.ax_fp = self.fig.add_subplot(1,2,2,projection='polar')
        lim=self.radius_max+1
        self.ax_top.set_xlim(-lim,lim); self.ax_top.set_ylim(-lim,lim); self.ax_top.set_aspect('equal')
        self.ax_top.set_title('Top-down View')
        self.ax_fp.set_title('First-person View'); self.ax_fp.set_theta_zero_location('N')
        self.ax_fp.set_theta_direction(-1); self.ax_fp.set_thetamin(-np.degrees(self.fov)/2)
        self.ax_fp.set_thetamax(np.degrees(self.fov)/2); self.ax_fp.set_rmax(self.fp_dist_max)
        plt.ion(); plt.show()

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        if mode!='human': return
        if not hasattr(self,'fig') or not plt.fignum_exists(self.fig.number): self._init_render()
        # top-down
        lim=self.radius_max+1; self.ax_top.clear(); self.ax_top.set_xlim(-lim,lim); self.ax_top.set_ylim(-lim,lim); self.ax_top.set_aspect('equal')
        self.ax_top.plot(self.agent_pos[0],self.agent_pos[1],'bo')
        dN,dE=np.cos(self.agent_heading),np.sin(self.agent_heading)
        self.ax_top.arrow(self.agent_pos[0],self.agent_pos[1],dE,dN,head_width=0.3,length_includes_head=True,color='b')
        self.ax_top.plot(self.target_pos[0],self.target_pos[1],'r*')
        self.ax_top.text(-10.0,-10.0,str(self.raw_vf))
        self.ax_top.text(-10.0,-12.0,str(self.raw_vr))
        self.ax_top.text(-10.0,-14.0,str(self.raw_w))
        circ = patches.Circle(self.target_pos, radius=self.goal_radius, 
                          fill=False, edgecolor='purple', linewidth=2)
        self.ax_top.add_patch(circ)

        # first-person
        self.ax_fp.clear()
        for angle in (-self.fov/2,self.fov/2): self.ax_fp.plot([angle,angle],[0,self.fp_dist_max],'k--')
        delta=self.target_pos-self.agent_pos; a2t=self.theta
        rct=np.linalg.norm(delta)
        self.ax_fp.text(0.1,0.1,str(self.theta))
        if abs(a2t)<=self.fov/2: self.ax_fp.plot(a2t,rct,'r*')
        circ_fp = plt.Circle((0,0), radius=self.goal_radius, transform=self.ax_fp.transData._b,
                         fill=False, edgecolor='purple', linewidth=2)
        self.ax_fp.add_patch(circ_fp)
        plt.pause(0.001)
