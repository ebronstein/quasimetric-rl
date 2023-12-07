# Value Functions for Goal-Conditioned Reinforcement Learning

**Authors**
- Eli Bronstein (26086997)
- Kevin Frans (3033673452)
- Toby Kreiman (3034607850)
- Tara Sadjadpour (3032788592)
- Zhiyuan (Paul) Zhou (3034663321)

## Requirements

The code has been tested on CUDA 11.8.

Software dependencies: see `requirements.txt`.

## Code structure

- `quasimetric_rl.modules` implements the actor and critic components, as well as their associated QRL losses.
- `quasimetric_rl.data` implements data loading and memory buffer utilities, as well as creation of environments.
- `online/`: online setting.
  - `run_gcrl.sh`: script to run online GCRL experiments.
  - `main.py`: entry point for online experiments.
- `offline/`: offline setting.
  - `main.py` entry point for training the offline experiments.
  - `eval_policy.py`: evaluates the learned policy.
  - `check_value_func.py`: computes value function errors.
  - `visualize_maze_value.py` and `visualize_values_for_maze_exps.py`: visualizes heatmap of value function for maze environments.
- `plots.ipynb`: Jupyter notebook to generate plots/figures.
- `fig`: figures.
- `scripts`: convenience scripts (e.g., for training online/offline experiments).
- `wandb`: Results exported from Wandb.
