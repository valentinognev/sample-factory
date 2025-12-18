"""
Enjoy script for the drone circle environment.

After training, evaluate the policy by running:
python -m sf_examples.enjoy_drone_circle_env --algo=APPO --env=drone_circle_env --experiment=drone_circle

Options:
    --no_render: Disable visualization
    --fps: Frames per second for rendering (default: 10)
    --save_video: Save video to file
"""

import sys

from sample_factory.enjoy import enjoy
from sf_examples.train_drone_circle_env import parse_custom_args, register_custom_components


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args(evaluation=True)
    
    # Set render mode if not specified
    if not hasattr(cfg, 'no_render') or not cfg.no_render:
        # Enable rendering by default
        pass
    
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
