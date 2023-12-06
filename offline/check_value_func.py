import argparse
import os
import pickle
import re
import yaml

import torch
from omegaconf import OmegaConf, SCMode
from tqdm import tqdm

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf
from .eval_utils import get_checkpoint_dict, get_exp_paths


def main(args):
    paths = args.paths
    ckpt = args.ckpt
    regex = args.regex
    dry_run = args.dry_run

    exp_paths = get_exp_paths(paths)
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
        check_value(exp_dir, which_ckpt=ckpt)


def check_value(
    exp_dir: str, which_ckpt: str = "latest", num_episodes: int = 100
):
    if which_ckpt not in ["latest", "all"]:
        raise ValueError(f"Invalid which_ckpt: {which_ckpt}")

    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"config file not found: {config_path}. Skipping.")
        return

    with open(config_path) as f:
        conf = OmegaConf.create(yaml.safe_load(f))

    # Load the dataset.
    dataset: Dataset = Dataset.Conf(
        kind=conf.env.kind, name=conf.env.name
    ).make()
    all_episodes = dataset.load_episodes()
    device_str = f"{conf.device.type}:{conf.device.index}"
    all_observations = []
    for i, episode in enumerate(all_episodes):
        if i >= num_episodes:
            break
        all_observations.append(episode.all_observations.to(device_str))

    ckpts = get_checkpoint_dict(exp_dir)
    if which_ckpt == "latest":
        epoch, it = max(ckpts.keys())
        ckpts = {(epoch, it): ckpts[epoch, it]}

    out = []

    for (epoch, it), checkpoint in tqdm(ckpts.items()):
        print("Evaluating epoch", epoch, "iter", it)

        # Make the agent.
        agent_conf: QRLConf = OmegaConf.to_container(
            OmegaConf.merge(
                OmegaConf.structured(QRLConf()), conf.agent
            ),  # overwrite with loaded conf
            structured_config_mode=SCMode.INSTANTIATE,  # create the object
        )
        agent: QRLAgent = agent_conf.make(
            env_spec=dataset.env_spec, total_optim_steps=1
        )[0]

        agent = agent.to(device_str)

        # Load checkpoint.
        agent.load_state_dict(torch.load(checkpoint)["agent"])
        actor = agent.actor
        if actor is None:
            print(f"Actor is None for checkpoint {checkpoint}. Skipping.")
            continue

        actor.eval()

        for k in tqdm(range(1, 10)):
            print("k =", k)
            episode_values = []
            for observations in all_observations:
                if len(observations) < k:
                    continue
                start_observations = observations[:-k]
                end_observations = observations[k:]
                with torch.no_grad():
                    values = (
                        agent.critics[0](start_observations, end_observations)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                episode_values.append(values)

            # print("mean:", np.concatenate(episode_values).mean())

            out.append(
                {"epoch": epoch, "iter": it, "k": k, "values": episode_values}
            )

    # Save data to pickle file.
    out_path = os.path.join(exp_dir, "value_func.pkl")
    print("Saving to", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)


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
