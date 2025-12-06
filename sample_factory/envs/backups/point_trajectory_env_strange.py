import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os, atexit, time
from datetime import datetime

from sample_factory.envs.env_utils import TrainingInfoInterface


def _wrap_angle(a: float) -> float:
    # wrap to [-pi, pi]
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class PointTrajectoryEnv(gym.Env, TrainingInfoInterface):
    """
    2D drone pursues a moving target.

    Actions (3):
        [v_f_cmd, v_r_cmd, w_cmd]  in [-1,1]
        -> scaled to [±v_f_max, ±v_r_max, ±w_max], applied via 1st-order response (tau_v, tau_w)

    Observations (4):
        [ r, theta, theta_dot, r_dot ]
        r         : range to target (m)
        theta     : target bearing in body frame [-pi,pi] (rad)
        theta_dot : d(theta)/dt (rad/s)
        r_dot     : radial closing rate along body-forward (m/s)

    Target modes:
        - "random": OU velocity in EN plane with clamp
        - "circle": constant tangential speed around origin (radius = circle_radius)
        - "square": constant speed along a square loop (side = square_side)

    Rendering:
        - Top-down EN on the left, first-person polar on the right
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, full_env_name: str, cfg=None, env_config=None, render_mode=None):
        super().__init__()
        self.full_env_name = full_env_name
        self.cfg = cfg or type("cfg", (), {})()
        self.env_config = env_config if isinstance(env_config, dict) else {}

        # ---------- Core params ----------
        self.dt        = float(self.env_config.get("dt", 0.1))
        self.max_steps = int(self.env_config.get("max_steps", 1000))
        self.world_r   = float(self.env_config.get("world_radius", 25.0))

        # Drone limits and dynamics
        self.v_f_max = float(self.env_config.get("v_f_max", 2.5))
        self.v_r_max = float(self.env_config.get("v_r_max", 2.5))
        self.w_max   = float(self.env_config.get("w_max",  2.0))
        self.tau_v   = float(self.env_config.get("tau_v",  0.4))
        self.tau_w   = float(self.env_config.get("tau_w",  0.4))
        self.psi     = float(self.env_config.get("heading", 0.0))   # initial yaw (rad)

        # Target OU velocity params (random mode)
        self.theta_t   = float(self.env_config.get("tgt_ou_theta", 0.5))
        self.sigma     = float(self.env_config.get("tgt_ou_sigma", 1.3))
        self.tgt_v_max = float(self.env_config.get("tgt_v_max",   4.0))

        # Target deterministic trajectory params
        self.trajectory_mode = str(self.env_config.get("trajectory", "random")).lower()
        if self.trajectory_mode not in ("random", "circle", "square", "waypoint"):
            self.trajectory_mode = "random"

        # circle (uniform tangential speed)
        self.circle_radius = float(self.env_config.get("circle_radius", 8.0))
        self.circle_speed  = float(self.env_config.get("circle_speed",  4.5))
        self._circle_theta = 0.0  # internal angle state

        # square (uniform speed along edges)
        self.square_side    = float(self.env_config.get("square_side", 12.0))
        self.square_speed   = float(self.env_config.get("square_speed",  3.0))
        self._sq_edge       = 0   # 0..3 (bottom, right, top, left)
        self._sq_s          = 0.0 # progress along current edge in [0, L]
        self._sq_dir        = +1  # +1 = CCW around square

        # waypoint
        self.waypoint = [10.0, 0.0]

        # Reward shaping & capture
        self.w_r      = float(self.env_config.get("w_r", 1.0))
        self.w_th     = float(self.env_config.get("w_theta", 0.5))
        self.w_act    = float(self.env_config.get("w_act", 0.01))  # currently multiplied by 0.0
        self.r_gate0  = float(self.env_config.get("theta_gate_r0", 1.0))
        self.r_gate1  = float(self.env_config.get("theta_gate_r1", 5.0))
        self.gate_min = float(self.env_config.get("theta_gate_min", 0.1))
        self.cap_r    = float(self.env_config.get("capture_radius", 0.5))
        self.cap_R    = float(self.env_config.get("capture_bonus", 5.0))

        # FPV display params
        self.fov      = float(self.env_config.get("fpv_fov", 2 * np.pi / 3))
        self.fpv_rmax = float(self.env_config.get("fpv_rmax", 20.0))

        # Spaces
        obs_high = np.array(
            [ self.world_r, np.pi, 5.0, self.v_f_max + self.v_r_max ],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32), dtype=np.float32)

        # States
        self.agent_pos = np.zeros(2, dtype=np.float32)  # [E, N]
        self.agent_v   = np.zeros(2, dtype=np.float32)
        self.v_body    = np.zeros(2, dtype=np.float32)  # filtered body velocities [fwd, right]
        self.w_yaw     = 0.0
        self.target_pos= np.zeros(2, dtype=np.float32)
        self.v_tgt     = np.zeros(2, dtype=np.float32)

        self.prev_theta = 0.0

        # Rendering
        self.auto_reset_on_timeout = bool(self.env_config.get("auto_reset_on_timeout", False))
        self.render_mode = render_mode
        self._fig = None
        self._ax_top = None
        self._ax_fpv = None
        self._last_action = np.zeros(3, dtype=np.float32)
        self._last_obs    = np.zeros(4, dtype=np.float32)


        # Logging
        log_dir  = self.env_config.get("log_dir", ".")
        os.makedirs(log_dir, exist_ok=True)
        log_name = f"pt_{datetime.now().strftime('%y%m%d_%H%M%S')}.csv"
        self._log = open(os.path.join(log_dir, log_name), "w")
        self._log.write("t,E,N,psi,tgtE,tgtN,vf,vr,wyaw,vE_tgt,vN_tgt,reward\n")
        atexit.register(lambda: self._log.close() if self._log and not self._log.closed else None)

        # RNG
        self.np_random = np.random.default_rng()

    # ---------- Observations ----------
    def _obs(self):
        delta = self.target_pos - self.agent_pos
        r = float(np.hypot(delta[0], delta[1]))
        theta_world = np.arctan2(delta[0], delta[1])       # atan2(E, N)
        theta_body  = _wrap_angle(theta_world - self.psi)  # [-pi, pi]

        theta_dot = _wrap_angle(theta_body - self.prev_theta) / max(1e-8, self.dt)
        self.prev_theta = theta_body

        rel_v = self.v_tgt - self.agent_v
        fwd = np.array([np.sin(self.psi), np.cos(self.psi)], dtype=np.float32)  # world unit along body-forward
        r_dot = float(np.dot(rel_v, fwd))

        return np.array([r, theta_body, theta_dot, r_dot], dtype=np.float32)

    # ---------- Target motion ----------
    def _step_target(self):
        if self.trajectory_mode == "circle":
            R = max(1e-6, self.circle_radius)
            omega = self.circle_speed / R
            self._circle_theta = (self._circle_theta + omega * self.dt) % (2*np.pi)
            E = R * np.sin(self._circle_theta)
            N = R * np.cos(self._circle_theta)
            vE =  self.circle_speed * np.cos(self._circle_theta)
            vN = -self.circle_speed * np.sin(self._circle_theta)
            self.target_pos[:] = [E, N]
            self.v_tgt[:] = [vE, vN]
            return

        elif self.trajectory_mode == "square":
            L = max(1e-6, self.square_side)
            ds = self.square_speed * self.dt
            self._sq_s += ds

            # advance edges as needed
            while self._sq_s >= L - 1e-12:
                self._sq_s -= L
                self._sq_edge = (self._sq_edge + self._sq_dir) % 4

            s = self._sq_s
            # square centered at origin; start at (-L/2, -L/2) bottom-left, go CCW
            half = L / 2.0
            if self._sq_edge == 0:   # bottom: (-half,-half) -> ( half,-half)
                E, N = (-half + s, -half)
                vE, vN = ( self.square_speed, 0.0)
            elif self._sq_edge == 1: # right: ( half,-half) -> ( half, half)
                E, N = ( half, -half + s)
                vE, vN = (0.0, self.square_speed)
            elif self._sq_edge == 2: # top:   ( half, half) -> (-half, half)
                E, N = ( half - s,  half)
                vE, vN = (-self.square_speed, 0.0)
            else:                    # left:  (-half, half) -> (-half,-half)
                E, N = (-half,  half - s)
                vE, vN = (0.0, -self.square_speed)

            self.target_pos[:] = [E, N]
            self.v_tgt[:] = [vE, vN]
            return

        elif self.trajectory_mode == "waypoint":
            E, N = self.waypoint
            self.target_pos[:] = [E, N]
            self.v_tgt[:] = [0, 0]

        # --- random / OU ---
        else:
            noise = self.np_random.standard_normal(2).astype(np.float32)
            self.v_tgt += (-self.theta_t * self.v_tgt) * self.dt + self.sigma * np.sqrt(self.dt) * noise
            spd = float(np.hypot(self.v_tgt[0], self.v_tgt[1]))
            if spd > self.tgt_v_max:
                self.v_tgt *= (self.tgt_v_max / (spd + 1e-8))
            self.target_pos += self.v_tgt * self.dt

    # ---------- Reward ----------
    def _theta_gate(self, r: float) -> float:
        if r <= self.r_gate0:
            return self.gate_min
        if r >= self.r_gate1:
            return 1.0
        t = (r - self.r_gate0) / (self.r_gate1 - self.r_gate0 + 1e-8)
        return self.gate_min + (1.0 - self.gate_min) * t

    def _reward(self, obs, action) -> float:
        r, theta, theta_dot, r_dot = obs
        gate = self._theta_gate(r)
        pos_term   = -self.w_r  * (r * r)
        angle_term = -self.w_th * (1.0 - np.cos(theta)) * gate
        act_term   = -0.0       * self.w_act * float(np.dot(action, action))  # keep zeroed unless you want action penalty
        bonus      = self.cap_R if r <= self.cap_r else 0.0
        return pos_term + angle_term + act_term + bonus

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.step_count = 0

        # Agent initial pose near origin
        E = np.random.uniform(-self.world_r,self.world_r)
        N = np.random.uniform(-self.world_r,self.world_r)
        self.agent_pos[:] = [E,N]
        self.psi = _wrap_angle(float(self.env_config.get("heading", 0.0)) +
                               self.np_random.uniform(-0.05, 0.05))
        self.v_body[:] = 0.0
        self.w_yaw = 0.0
        self.agent_v[:] = 0.0

        # Target initial state
        if self.trajectory_mode == "circle":
            self._circle_theta = float(self.np_random.uniform(0, 2*np.pi))
            E = self.circle_radius * np.sin(self._circle_theta)
            N = self.circle_radius * np.cos(self._circle_theta)
            self.target_pos[:] = [E, N]
            self.v_tgt[:] = 0.0

        elif self.trajectory_mode == "square":
            self._sq_edge = 0
            self._sq_s = 0.0
            self._sq_dir = +1
            self.target_pos[:] = [-self.square_side/2.0, -self.square_side/2.0]
            self.v_tgt[:] = [self.square_speed, 0.0]

        elif self.trajectory_mode == "waypoint":
            E = np.random.uniform(-self.world_r,self.world_r)
            N = np.random.uniform(-self.world_r,self.world_r)
            self.waypoint = [E,N]
            self.target_pos[:] = self.waypoint
            self.v_tgt[:] = [0.0, 0.0]

        else:  # "random"
            ang  = self.np_random.uniform(0, 2*np.pi)
            rad  = self.np_random.uniform(0.3*self.world_r, 0.8*self.world_r)
            self.target_pos[:] = [rad*np.sin(ang), rad*np.cos(ang)]
            self.v_tgt[:] = self.np_random.normal(0.0, 0.3, size=2).astype(np.float32)

        # prime prev_theta
        delta = self.target_pos - self.agent_pos
        self.prev_theta = _wrap_angle(np.arctan2(delta[0], delta[1]) - self.psi)

        if self.render_mode == "human":
            self._init_render()

        obs0 = self._obs()
        self._last_obs[:] = obs0
        self._last_action[:] = 0.0
        return obs0, {}
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # 1) first-order response for body velocities and yaw rate
        v_f_cmd = float(action[0]) * self.v_f_max
        v_r_cmd = float(action[1]) * self.v_r_max
        w_cmd   = float(action[2]) * self.w_max

        self.v_body += (np.array([v_f_cmd, v_r_cmd], dtype=np.float32) - self.v_body) * (self.dt / max(1e-6, self.tau_v))
        self.w_yaw  += (w_cmd - self.w_yaw) * (self.dt / max(1e-6, self.tau_w))

        # integrate yaw
        self.psi = _wrap_angle(self.psi + self.w_yaw * self.dt)

        # convert body vel to world EN
        c = np.cos(self.psi); s = np.sin(self.psi)
        R_b2w = np.array([[ s,  c],
                          [ c, -s]], dtype=np.float32)  # (E,N) from (fwd,right) with CW heading
        vel_world = R_b2w @ self.v_body

        # integrate agent position & world velocity for r_dot calc
        self.agent_pos += vel_world * self.dt
        self.agent_v[:] = vel_world

        # 2) target evolution
        self._step_target()

        # 3) obs + reward
        obs = self._obs()
        self._last_action[:] = action
        self._last_obs[:] = obs
        reward = self._reward(obs, action)

        # 4) log
        self._log.write(f"{time.time():.3f},{self.agent_pos[0]:.3f},{self.agent_pos[1]:.3f},{self.psi:.3f},"
                        f"{self.target_pos[0]:.3f},{self.target_pos[1]:.3f},"
                        f"{self.v_body[0]:.3f},{self.v_body[1]:.3f},{self.w_yaw:.3f},"
                        f"{self.v_tgt[0]:.3f},{self.v_tgt[1]:.3f},{reward:.5f}\n")

        # 5) done flags (auto-reset only if enabled)
        self.step_count += 1
        is_timeout = self.step_count >= self.max_steps

        info = {"r": float(obs[0]), "theta": float(obs[1])}

        if is_timeout and self.auto_reset_on_timeout:
            # Return the final (timeout) transition and immediately reset for enjoy
            final_obs = obs
            final_rew = float(reward)
            final_info = dict(info)
            final_info["TimeLimit.truncated"] = True

            self.reset()
            if self.render_mode == "human":
                self.render()

            return final_obs, final_rew, False, True, final_info

        # Training (or enjoy without flag): standard time-limit semantics
        terminated = False
        truncated  = is_timeout
        if is_timeout:
            info["TimeLimit.truncated"] = True

        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    # ---------- Rendering ----------
    def _init_render(self):
        import matplotlib.pyplot as plt
        if self._fig is not None and plt.fignum_exists(self._fig.number):
            return
        self._fig = plt.figure(figsize=(10, 5))
        self._ax_top = self._fig.add_subplot(1, 2, 1)
        self._ax_fpv = self._fig.add_subplot(1, 2, 2, projection='polar')
        self._fig.tight_layout()

    def render(self):
        import matplotlib.pyplot as plt
        self._init_render()

        # Top-down
        ax = self._ax_top
        ax.clear()
        ax.set_title("Top-Down EN")
        R = self.world_r
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.3)
        circ = plt.Circle((0, 0), self.cap_r, edgecolor='g', facecolor='none', alpha=0.6)
        ax.add_patch(circ)
        # agent position and heading arrow
        ax.plot(self.agent_pos[0], self.agent_pos[1], 'bo', label='Agent')
        dN, dE = np.cos(self.psi), np.sin(self.psi)
        ax.arrow(self.agent_pos[0], self.agent_pos[1], dE, dN,
                 head_width=0.4, length_includes_head=True, alpha=0.85)
        # target
        ax.plot(self.target_pos[0], self.target_pos[1], 'r*', label='Target')
        # ---------- Text overlays on Top-Down ----------
        act = self._last_action
        ob  = self._last_obs
        act_str = (
            f"actions\n"
            f"v_f: {act[0]:.2f}\n"
            f"v_r: {act[1]:.2f}\n"
            f" w : {act[2]:.2f}"
        )
        obs_str = (
            f"obs\n"
            f" r: {ob[0]:.2f}\n"
            f"θ : {ob[1]:.2f}\n"
            f"θ̇: {ob[2]:.2f}\n"
            f" ṙ: {ob[3]:.2f}"
        )
        # bottom-left: actions
        ax.text(
            0.02, 0.02, act_str, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
        )
        # bottom-right: observations
        ax.text(
            0.98, 0.02, obs_str, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
        )
        # FPV polar
        fp = self._ax_fpv
        fp.clear()
        fp.set_title("First-Person View (θ,r)")
        fp.set_theta_zero_location('N')
        fp.set_theta_direction(-1)
        fp.set_thetamin(-np.degrees(self.fov)/2)
        fp.set_thetamax( np.degrees(self.fov)/2)
        fp.set_rmax(self.fpv_rmax)
        fp.grid(True)

        delta = self.target_pos - self.agent_pos
        r = float(np.hypot(delta[0], delta[1]))
        theta_world = np.arctan2(delta[0], delta[1])
        theta_body  = _wrap_angle(theta_world - self.psi)

        fp.plot([ -self.fov/2, -self.fov/2 ], [0, self.fpv_rmax], 'k--', linewidth=1)
        fp.plot([  self.fov/2,  self.fov/2 ], [0, self.fpv_rmax], 'k--', linewidth=1)

        if abs(theta_body) <= self.fov/2:
            fp.plot(theta_body, r, 'r*')
        else:
            edge_theta = np.sign(theta_body) * (self.fov/2)
            fp.plot(edge_theta, min(r, self.fpv_rmax), 'r*')

        th = np.linspace(-self.fov/2, self.fov/2, 200)
        fp.plot(th, np.full_like(th, self.cap_r), 'g-', alpha=0.6)

        plt.pause(0.001)

    def close(self):
        try:
            if self._log and not self._log.closed:
                self._log.close()
        except Exception:
            pass


def make_point_env(full_env_name: str, cfg=None, env_config=None, render_mode=None):
    return PointTrajectoryEnv(full_env_name, cfg=cfg, env_config=env_config, render_mode=render_mode)
