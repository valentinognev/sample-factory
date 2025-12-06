import sys
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from datetime import datetime

# parse and strip custom flags (record_video, render_mode, trajectory)
_custom_parser = argparse.ArgumentParser(add_help=False)
_custom_parser.add_argument('--record_video', action='store_true',
                            help='Record the enjoy rendering to a video file')
_custom_parser.add_argument('--render_mode', type=str, default=None,
                            help='Override render mode for enjoy')
_custom_parser.add_argument('--trajectory', type=str,
                            choices=['circle', 'square', 'waypoint', 'waypoints', 'hold', 'random'],
                            help='Preset target trajectory')
_custom_parser.add_argument('--hold_point', type=str, default=None,
                            help='Hold point as "x,y" for waypoint/hold mode')
_custom_args, _remaining = _custom_parser.parse_known_args()

# rebuild sys.argv without our custom flags so SF parser doesn't choke
_new_argv = [sys.argv[0]]
skip_next = False
for arg in sys.argv[1:]:
    if skip_next:
        skip_next = False
        continue
    # strip our custom flags
    if arg in ('--record_video', '--save_video'):
        continue
    if arg.startswith('--render_mode'):
        if '=' not in arg:
            skip_next = True
        continue
    if arg == '--trajectory' or arg == '--hold_point':
        skip_next = True
        continue
    if arg.startswith('--trajectory=') or arg.startswith('--hold_point='):
        continue
    _new_argv.append(arg)
sys.argv = _new_argv

from train_point_trajectory import parse_args, register_point_env
from sample_factory.enjoy import enjoy as sf_enjoy


def main():
    """Script entry-point for evaluation with automatic video naming and preset trajectories."""
    register_point_env()
    cfg = parse_args(evaluation=True)
    # satisfy SF enjoy argument requirements
    cfg.cli_args = vars(cfg)
    if not hasattr(cfg, 'eval_env_frameskip'):
        cfg.eval_env_frameskip = None
    if not hasattr(cfg, 'env_frameskip'):
        cfg.env_frameskip = None
    cfg.auto_reset_on_timeout = True
    
    # apply preset trajectory if provided
    if _custom_args.trajectory:
        cfg.trajectory = _custom_args.trajectory

    # apply hold_point if provided
    if _custom_args.hold_point:
        try:
            x_str, y_str = _custom_args.hold_point.split(',')
            cfg.hold_point = [float(x_str), float(y_str)]
        except Exception:
            cfg.hold_point = [0.0, 0.0]

    # override render_mode if provided
    if _custom_args.render_mode:
        cfg.render_mode = _custom_args.render_mode

    record = _custom_args.record_video

    if record:
        # determine FPS
        fps = getattr(cfg, 'fps', None) or getattr(cfg, 'render_fps', None) or 30
        fps = int(fps)
        # prepare directory for logs
        log_dir = os.path.join(getattr(cfg, 'experiment_dir', '.'), 'enjoy_videos')
        os.makedirs(log_dir, exist_ok=True)
        # auto-generate filename with timestamp
        ts = datetime.now().strftime('%d_%m_%y-%H_%M_%S')  # format: DD_MM_YY-HHMMSS
        out_path = os.path.join(log_dir, f'record_{ts}.mp4')
        print(f"Recording video to {out_path} at {fps} FPS")

        writer_holder = {'writer': None}
        orig_pause = plt.pause
        def pause_and_grab(interval):
            orig_pause(interval)
            fig = plt.gcf()
            # setup writer on first frame
            if writer_holder['writer'] is None:
                FFMpegWriter = manimation.writers['ffmpeg']
                writer = FFMpegWriter(fps=fps, metadata={'artist': 'RL'}, bitrate=1800)
                writer.setup(fig, out_path, dpi=100)
                writer_holder['writer'] = writer
            writer_holder['writer'].grab_frame()

        try:
            plt.pause = pause_and_grab
            status = sf_enjoy(cfg)
        finally:
            plt.pause = orig_pause
            if writer_holder['writer'] is not None:
                writer_holder['writer'].finish()
    else:
        status = sf_enjoy(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
