import argparse
import os
import re
import yaml
from typing import List

from omegaconf import OmegaConf, SCMode

from .main import Conf
from .visualize_maze_value import visualize_value


def _is_exp_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "config.yaml")
    )


def _get_exp_paths(paths: List[str]) -> List[str]:
    # Walk through the paths and find the experiment directories.
    exp_paths = []
    for path in paths:
        if _is_exp_dir(path):
            exp_paths.append(path)
        else:
            for root, dirs, _ in os.walk(path):
                for dir in dirs:
                    subdir = os.path.join(root, dir)
                    if _is_exp_dir(subdir):
                        exp_paths.append(subdir)
    return exp_paths


def main(args):
    paths = args.paths
    ckpt = args.ckpt
    regex = args.regex
    dry_run = args.dry_run

    exp_paths = _get_exp_paths(paths)
    if not exp_paths:
        raise ValueError(f"No experiment directories found in {paths}")

    # Filter exp_paths by regex.
    if regex:
        regex = re.compile(regex)
        exp_paths = list(filter(regex.search, exp_paths))
        if not exp_paths:
            raise ValueError(
                f"No experiment directories found in {paths} matching {regex}"
            )

    if dry_run:
        print("Dry-run mode. Will not save anything.")
        print("Experiment directories:")
        for exp_dir in exp_paths:
            print(f"  {exp_dir}")
        print(f"Checkpoint option: {ckpt}")
        return

    for exp_dir in exp_paths:
        with open(os.path.join(exp_dir, "config.yaml"), "r") as f:
            conf = OmegaConf.create(yaml.safe_load(f))

        conf: Conf = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.structured(Conf()), conf),
            structured_config_mode=SCMode.INSTANTIATE,  # create the object
        )
        conf.set_output_folder()

        visualize_value(conf, exp_dir, which_ckpt=ckpt)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process CLI args.")

    parser.add_argument("paths", nargs="+", help="Experiment paths.")
    parser.add_argument(
        "-c",
        "--ckpt",
        choices=["latest", "all"],
        default="latest",
        help="Checkpoint option (latest or all)",
    )
    parser.add_argument(
        "-r",
        "--regex",
        default="",
        help="Regex pattern for experiment directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
