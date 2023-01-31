import torch
import numpy as np
from tqdm import tqdm
from model.ppo import PPO
from environment.pedsim import Pedsim
from utils.utils import get_args, set_seed, pack_state, mod2pi
from utils.metrics import calc_TEC, find_CAP


if __name__ == '__main__':
    ARGS = get_args(
        ENV=dict(type=str, default='./dataset/GC/GC_Dataset_ped1-12685_time1560-1620_interp9_xrange5-25_yrange15-35.npy'),
        # ENV=dict(type=str, default='dataset\UCY\UCY_Dataset_time162-216_timeunit0.08.npy'),
    )
    set_seed(ARGS.SEED)

    # init model
    model = PPO(ARGS).to(ARGS.DEVICE)
    model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))

    # load dataset
    meta_data, trajectoris, des, obs = np.load(ARGS.ENV, allow_pickle=True)
    N = len(trajectoris)
    T = np.max([t for traj in trajectoris for x, y, t in traj]) + 1
    position = torch.full((N, T, 2), torch.nan, device=ARGS.DEVICE)
    for p, traj in enumerate(trajectoris):
        for x, y, t in traj:
            position[p, t, 0] = x
            position[p, t, 1] = y
    nan_flag = position.isnan().any(dim=-1)  # (N, T)
    into_env = (~nan_flag) & (nan_flag.roll(1, 1))  # (N, T)
    exit_env = (nan_flag) & (~nan_flag.roll(1, 1))  # (N, T)
    into_env[nan_flag.logical_not().all(dim=-1), 0] = True
    exit_env[nan_flag.logical_not().all(dim=-1), 0] = True
    assert (into_env.sum(dim=1) == 1).all() and (exit_env.sum(dim=1) == 1).all(), "A pedestrian enter the env for more/less than 1 times!"
    time = torch.arange(T, device=ARGS.DEVICE)
    into_time = torch.masked_select(time, into_env)  # (N,)
    exit_time = torch.masked_select(time, exit_env)  # (N,)
    exit_time[exit_time == 0] = T

    velocity = position.diff(dim=1, prepend=position[:, (0,), :]) / meta_data['time_unit']
    velocity[:, into_time, :] = velocity.roll(-1, 1)[:, into_time, :]
    one_step_flag = (into_time + 1 == exit_time)
    velocity[one_step_flag, into_time[one_step_flag], :] = 0.

    destination = torch.FloatTensor(des)[:, 0, :2]  # (N, 2)
    obstacle = torch.FloatTensor(obs)

    env_real = Pedsim(ARGS)
    env_real.init_ped(position, velocity, destination)

    
    # evaluation
    CAP_flag = find_CAP(env_real)
    CAP_start = CAP_flag.clone(); CAP_start[:, :, 1:] &= ~CAP_start[:, :, :-1]
    CAP_final = CAP_flag.clone(); CAP_final[:, :, :-1] &= ~CAP_final[:, :, 1:]

    TECs = []
    Collisions = []
    pbar = tqdm(range(T))
    for t in pbar:
        if not CAP_start[:, :, t].any(): continue

        # simulate from the state at time t        
        env_imit = Pedsim(env_real.args)
        env_imit.meta_data = env_real.meta_data
        env_imit.add_pedestrian(env_real.position[:, t, :], env_real.velocity[:, t, :], env_real.destination, init=True)
        for s in range(t + 1, min(t + 101, T)):
            mask = env_imit.mask[:, -1] & ~env_imit.arrive_flag[:, -1]
            action = torch.full((env_imit.num_pedestrians, 2), torch.nan, device=env_imit.device)
            if mask.any():
                action[mask, :], _ = model(pack_state(*env_imit.get_state())[mask])
            env_imit.action(action[:, 0], action[:, 1], enable_nan_action=True)
            into_flag = env_real.mask[:, s] & ~env_real.mask[:, s - 1]
            if into_flag.any():
                env_imit.position[into_flag, -1, :] = env_real.position[into_flag, s, :]
                env_imit.velocity[into_flag, -1, :] = env_real.velocity[into_flag, s, :]
                env_imit.direction[into_flag, -1, :] = env_real.direction[into_flag, s, :]
                env_imit.destination[into_flag, :] = env_real.destination[into_flag, :]
                env_imit.arrive_flag[into_flag, -1] = ((env_real.position[into_flag, s, :] - env_real.destination[into_flag, :]).norm(dim=-1) < env_real.ped_radius)
                env_imit.mask[into_flag, -1] = True

        # evaluate the CAP
        for i, j in CAP_start[:, :, t].nonzero():
            view = mod2pi(env_imit.direction[i, :, 0] - torch.atan2(*(env_imit.position[j] - env_imit.position[i]).T.flip(0))).abs() < torch.pi / 2  # (T,)
            close = (env_imit.position[j] - env_imit.position[i]).norm(dim=-1) < 1.0  # (T,)
            naway = (((env_imit.position[j] - env_imit.position[i]) * (env_imit.velocity[j] - env_imit.velocity[i])).sum(dim=-1) / (env_imit.position[j] - env_imit.position[i]).norm(dim=-1).clamp(1e-8) < 0.5)  # (T,)
            valid = (view & naway) | close  # (T,)
            if not valid.any(): continue
            final = valid.nonzero()[-1, 0]
            if (final + 1 < env_imit.mask.shape[1]) and (~env_imit.mask[i, final + 1] or ~env_imit.mask[j, final + 1]): continue
            TEC = calc_TEC(env_imit.meta_data['time_unit'], env_imit.position[i, :final+1, :], env_imit.destination[(i,), :], lambda_E=4.73e-4, lambda_W=3.30e-3, lambda_M=6.17e-1)
            TECs.append(TEC.item())
            Collision = (env_imit.raw_velocity.norm(dim=-1) > env_imit.velocity.norm(dim=-1))[i, :final+1].sum()
            Collisions.append(Collision.item())

            pbar.set_postfix(dict(TEC=np.mean(TECs), Collision=np.mean(Collisions)))
    
    print(TECs, Collisions)    
    print(f'TEC = {np.mean(TECs)}, #Collision = {np.mean(Collisions)}')
