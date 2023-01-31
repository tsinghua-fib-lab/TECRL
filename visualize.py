"""
This script is used for visualization
- Rely on PyQt5 and pyqtgraph
- execute `python visualize.py --LOAD_MODEL <MODEL>` to visualize the performance of MODEL
    - For example, execute `python visualize.py --LOAD_MODEL ./checkpoint/demonstration/model_final.bin`
"""
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from environment.pedsim import Pedsim
from model.ppo import PPO
from utils.visualization import Visualization
from utils.utils import get_args, init_env, set_seed, pack_state

if __name__ == '__main__':
    ARGS = get_args()
    set_seed(ARGS.SEED)
    model = PPO(ARGS).to(ARGS.DEVICE)
    if ARGS.LOAD_MODEL is not None:
        model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))
    env = Pedsim(ARGS)
    init_env(env, ARGS)

    def update():
        with torch.no_grad():
            mask = env.mask[:, -1] & ~env.arrive_flag[:, -1]
            if not mask.any():
                return False
            action = torch.full((env.num_pedestrians, 2), torch.nan, device=env.device)
            state = pack_state(*env.get_state())
            action[mask, :], _ = model(state[mask], explore=True)
            env.action(action[:, 0], action[:, 1], enable_nan_action=True)
        return True
    env.update = update
    Visualization(env, model=model).play()
