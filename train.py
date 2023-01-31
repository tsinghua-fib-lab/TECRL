"""
This sciprt is used for train the model.
- Execute `python train.py`, the model will be saved in `./checkpoint/testproj/model_final.bin`
- See the function `get_args()` in `utils/utils.py` for default parameters.
"""
import os
import torch
import json
import logging
from model.ppo import PPO
from environment.pedsim import Pedsim
from utils.utils import init_env, get_args, set_seed, pack_state
from utils.visualization_cv import generate_gif


if __name__ == '__main__':
    # initialization
    ARGS = get_args()
    set_seed(ARGS.SEED)
    model = PPO(ARGS).to(ARGS.DEVICE)
    if ARGS.LOAD_MODEL is not None:
        model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))
    save_path = os.path.join(ARGS.SAVE_DIRECTORY, ARGS.UUID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with(open(os.path.join(save_path, f'args.log'), 'w')) as f:
        json.dump(ARGS.__dict__, f, indent=2)
    with(open(os.path.join(save_path, f'rewards.log'), 'w')) as f:
        f.write('REWARD ARRIVE ENERGY WORK COLL MENTAL\n')
    logging.getLogger().setLevel(logging.INFO)

    for episode in range(1, ARGS.MAX_EPISODES + 1):
        # generate environment
        env = Pedsim(ARGS)
        init_env(env, ARGS)

        # run & train
        reward, arrive_num, detail = model.run_episode(env, train=True)
        
        # log
        logging.info(f'[Epi{episode}] #Arrive: {arrive_num}, Reward: {reward:7.2f} [ {", ".join([f"{detail[r]:.1f}({r})" for r in detail])} ]')
        open(os.path.join(save_path, f'rewards.log'), 'a').write(f'{str(reward)} {" ".join([str(r) for r in detail.values()])}\n')
        
        # save (at epoch 100, 200, ..., 900, 1000, 2000, ..., 9000, ...)
        if episode >= 100 and str(episode)[1:] == '0' * len(str(episode)[1:]):
            # save model
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{episode}.bin'))
            # save GIF visualization
            for t in range(200):
                mask = env.mask[:, -1] & ~env.arrive_flag[:, -1]
                if not mask.any(): break
                action = torch.full((env.num_pedestrians, 2), torch.nan, device=env.device)
                action[mask, :], _ = model(pack_state(*env.get_state())[mask], explore=True)
                env.action(action[:, 0], action[:, 1], enable_nan_action=True)
            generate_gif(env, os.path.join(save_path, f'demo_{episode}.gif'))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_final.bin'))

        if episode % 10000 == 0:
            model.ARGS.ENTROPY /= 10.0