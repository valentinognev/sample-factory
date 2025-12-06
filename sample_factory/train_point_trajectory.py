import argparse, sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from envs.point_trajectory_env import PointTrajectoryEnv  # your renamed file

def make_point_env(full_env_name: str, cfg=None, env_config=None, render_mode=None):
    merged = dict(env_config or {})
    # pass through trajectory knobs (use existing names you support)
    for k in [
        "trajectory",
        "circle_radius", "circle_speed",
        "square_side", "square_speed",
        "dt", "tau_v", "tau_w",
        "v_f_max", "v_r_max", "w_max",
        "tgt_ou_theta", "tgt_ou_sigma", "tgt_v_max",
        "theta_gate_r0", "theta_gate_r1", "theta_gate_min",
        "capture_radius", "capture_bonus",
        "world_radius", "fpv_fov", "fpv_rmax",
        "max_steps", "heading",
        "auto_reset_on_timeout",
    ]:
        if hasattr(cfg, k):
            merged[k] = getattr(cfg, k)
    # parse hold_point "x,y" string to list if needed
    hp = merged.get("hold_point", None)
    if isinstance(hp, str):
        try:
            xs, ys = hp.split(",")
            merged["hold_point"] = [float(xs), float(ys)]
        except Exception:
            merged["hold_point"] = [0.0, 0.0]
    return PointTrajectoryEnv(full_env_name, cfg=cfg, env_config=merged, render_mode=render_mode)

def register_point_env():
    register_env("point_trajectory", make_point_env)

def add_point_env_args(env, parser: argparse.ArgumentParser, evaluation=False):
    parser.add_argument("--trajectory",
                        default="random",
                        type=str,
                        choices=["circle", "square", "random", "waypoints", "waypoint", "waypoint2", "hold"],
                        help="Target motion mode. 'waypoint/hold' = single fixed point, 'waypoints' = polyline.")

    # New, useful environment knobs
    g = parser.add_argument_group("PointTrajectoryEnv")
    g.add_argument("--dt", type=float, default=0.1)
    g.add_argument("--max_steps", type=int, default=100)
    g.add_argument("--world_radius", type=float, default=25.0)
    g.add_argument("--hold_point", type=str, default=None,
                   help='Hold point "x,y" for trajectory=waypoint/hold')

    # dynamics
    g.add_argument("--v_f_max", type=float, default=2.5)
    g.add_argument("--v_r_max", type=float, default=2.5)
    g.add_argument("--w_max", type=float, default=2.0)
    g.add_argument("--tau_v", type=float, default=0.4)
    g.add_argument("--tau_w", type=float, default=0.4)

    # target OU
    g.add_argument("--tgt_ou_theta", type=float, default=0.7)
    g.add_argument("--tgt_ou_sigma", type=float, default=1.0)
    g.add_argument("--tgt_v_max", type=float, default=2.0)

    # rewards & gates
    g.add_argument("--w_r", type=float, default=1.0)
    g.add_argument("--w_theta", type=float, default=0.25)
    g.add_argument("--w_act", type=float, default=0.01)
    g.add_argument("--theta_gate_r0", type=float, default=1.0)
    g.add_argument("--theta_gate_r1", type=float, default=5.0)
    g.add_argument("--theta_gate_min", type=float, default=0.1)
    g.add_argument("--capture_radius", type=float, default=0.5)
    g.add_argument("--capture_bonus", type=float, default=5.0)

def point_trajectory_override_defaults(env, parser: argparse.ArgumentParser):
    parser.set_defaults(
        encoder_mlp_layers=[64, 64, 64],
        learning_rate=5e-4,
        train_for_env_steps=1_000_000,
        env_frameskip=1,
    )

def parse_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_point_env_args(partial_cfg.env, parser, evaluation=evaluation)
    point_trajectory_override_defaults(partial_cfg.env, parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg

def main():
    register_point_env()
    cfg = parse_args()
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
