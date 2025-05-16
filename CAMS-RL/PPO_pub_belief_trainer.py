import csv
import os
import random
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import tyro
from omegaconf import OmegaConf
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from Gym_Envs.HexnerEnv_w_reward_n_belief import GymHexnerEnv
from algorithms.PPO.PPO_Pub_Belief import PPOAgent

# for logging csv
def log_to_csv(data, path):
    file_exists = False
    try:
        with open(path, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False

    num_envs: int = 128
    total_timesteps: int = 10_000_000
    num_steps: int = 64
    learning_rate: float = 2.5e-5
    anneal_lr: bool = True
    anneal_sigma: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.025
    vf_coef: float = 0.5
    max_grad_norm: float = 5
    save_interval: int = 5000
    # save_dir: str = "./saved_models/FINAL/PPO_SIMUL_CONV/1-step"
    save_dir: str = "./saved_models/FINAL_POLICY_TRAJ/"
    print_freq: int = 10

    dt: float = 1
    max_game_length = None
    ux_max: float = 12
    uy_max: float = 12
    dx_max: float = 12
    dy_max: float = 12
    p: float = 0.5
    include_action_history: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def make_env(dt, max_game_length, ux_max, uy_max, dx_max, dy_max, p, include_action_history, seed):
    def thunk():
        env = GymHexnerEnv(
            dt=dt,
            max_game_length=max_game_length,
            ux_max=ux_max,
            uy_max=uy_max,
            dx_max=dx_max,
            dy_max=dy_max,
            p=p,
            render_on=False,
            include_action_history=include_action_history
        )
        env.reset(seed=seed)
        return env
    return thunk

if __name__ == "__main__":
    args = tyro.cli(Args)

    # seed & determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # vectorized envs
    env_fns = [
        make_env(args.dt, args.max_game_length,
                 args.ux_max, args.uy_max,
                 args.dx_max, args.dy_max,
                 args.p, args.include_action_history,
                 args.seed + i)
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)
    tmp_env = envs.envs[0]
    max_game_length = tmp_env.max_game_length

    # dimensions
    state_dim       = 9 # (time, state)
    num_type_p1     = 2
    action_hist_dim = max_game_length * 4 if args.include_action_history else 0
    action_dim      = tmp_env.action_space["player1"].shape[0]

    # instantiate agents
    agent1 = PPOAgent(
        state_dim=state_dim,
        belief_dim=num_type_p1-1,
        action_hist_dim=action_hist_dim,
        action_dim=action_dim,
        num_type=num_type_p1,
        action_bounds=[args.ux_max, args.uy_max],
        hidden_sizes=[64, 64],
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        ppo_epochs=args.update_epochs,
        mini_batch_size=int((args.num_envs * args.num_steps) // args.num_minibatches),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.ent_coef,
        value_loss_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=device
    )
    agent2 = PPOAgent(
        state_dim=state_dim,
        belief_dim=num_type_p1-1,
        action_hist_dim=action_hist_dim,
        action_dim=action_dim,
        num_type=num_type_p1+1,  # dummy type for P2
        action_bounds=[args.dx_max, args.dy_max],
        hidden_sizes=[64, 64],
        has_private_type=False,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        ppo_epochs=args.update_epochs,
        mini_batch_size=int((args.num_envs * args.num_steps) // args.num_minibatches),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.ent_coef,
        value_loss_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=device
    )

    # initialize the weight and bias for the post head
    # with torch.no_grad():
    #     agent1.model.posts_head.weight.mul_(0)
    #     agent1.model.posts_head.bias.copy_(torch.tensor([0.5]))  # fix the bias


    # derived params
    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    # save config
    os.makedirs(args.save_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.create(vars(args)), os.path.join(args.save_dir, "config.yaml"))

    # storage buffers for P1
    obs_state_buf_p1 = torch.zeros((args.num_steps, args.num_envs, state_dim), device=device) # only physical state
    obs_type_buf_p1  = torch.zeros((args.num_steps, args.num_envs, num_type_p1), device=device)
    obs_belief_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    actions_buf_p1   = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    actions_buf_p2 = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logps_buf_p1     = torch.zeros((args.num_steps, args.num_envs), device=device)
    rews_buf_p1      = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf_p1     = torch.zeros((args.num_steps, args.num_envs), device=device)
    vals_buf_p1      = torch.zeros((args.num_steps, args.num_envs), device=device)

    # storage buffers for P2
    obs_state_buf_p2 = torch.zeros((args.num_steps, args.num_envs, state_dim), device=device)  # only physical state
    logps_buf_p2     = torch.zeros((args.num_steps, args.num_envs), device=device)
    rews_buf_p2      = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf_p2     = torch.zeros((args.num_steps, args.num_envs), device=device)
    vals_buf_p2      = torch.zeros((args.num_steps, args.num_envs), device=device)

    if args.include_action_history:
        obs_hist_buf_p1  = torch.zeros((args.num_steps, args.num_envs, action_hist_dim), device=device)
        obs_hist_buf_p2 = torch.zeros((args.num_steps, args.num_envs, action_hist_dim), device=device)

    # reset envs
    next_raw_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    # unpack next_obs for P1 and P2
    def unpack(obs_dict, player_key):
        arr = obs_dict[player_key]
        state = torch.tensor(arr["state"], dtype=torch.float32, device=device)
        belief = torch.tensor(arr["belief"], dtype=torch.float32, device=device)
        if "player_type" in arr:
            ptype = torch.tensor(arr["player_type"], dtype=torch.float32, device=device)
        else:
            ptype = None

        if args.include_action_history:
            hist = torch.tensor(arr["action_history"].reshape(args.num_envs, -1),
                                dtype=torch.float32, device=device)
        else:
            hist = None
        return state, ptype, belief, hist

    next_s1, next_t1, next_b, next_h1 = unpack(next_raw_obs, "player1")
    next_s2, next_t2, _, next_h2 = unpack(next_raw_obs, "player2")

    global_step = 0
    start_time  = time.time()
    # print_freq  = args.num_iterations // 10 if args.num_iterations // 10 < 100 else 100

    # for annealing sigma
    start_max, final_max = 0.0, -5.0

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # anneal lr
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            lrnow = frac * args.learning_rate
            for pg in agent1.optimizer.param_groups: pg["lr"] = lrnow
            for pg in agent2.optimizer.param_groups: pg["lr"] = lrnow

        # anneal log_std
        # ipdb.set_trace()
        if args.anneal_sigma:
            frac = (iteration - 1) / args.num_iterations
            cur_max = start_max + frac * (final_max - start_max)
            agent1.model.log_std_max.fill_(cur_max)
            agent2.model.log_std_max.fill_(cur_max)

        # anneal temp
        initial_temp, final_temp = 3.0, 1.0
        temperature = initial_temp - (initial_temp - final_temp) * ((iteration - 1) / args.num_iterations)
        temperature = max(temperature, final_temp)
        agent1.model.temp.fill_(temperature)
        agent2.model.temp.fill_(temperature)


        # rollout
        for step in range(args.num_steps):
            global_step += args.num_envs

            # store obs
            obs_state_buf_p1[step].copy_(next_s1)
            obs_type_buf_p1[step].copy_(next_t1)
            obs_belief_buf[step].copy_(next_b)
            obs_state_buf_p2[step].copy_(next_s2)
            dones_buf_p1[step].copy_(next_done)
            dones_buf_p2[step].copy_(next_done)

            if args.include_action_history:
                obs_hist_buf_p1[step].copy_(next_h1)
                obs_hist_buf_p2[step].copy_(next_h2)

            # build dicts
            if args.include_action_history:
                od1 = {"state": next_s1, "player_type": next_t1, "belief": next_b, "action_history": next_h1}
                od2 = {"state": next_s2, "belief": next_b, "action_history": next_h2}
            else:
                od1 = {"state": next_s1, "belief": next_b,"player_type": next_t1}
                od2 = {"state": next_s2, "belief": next_b}

            # get actions
            with torch.no_grad():
                a1, lp1, _, v1, b = agent1.model.get_action(od1)
                a2, lp2, _, v2, _ = agent2.model.get_action(od2)
                # ipdb.set_trace()
            actions_buf_p1[step].copy_(a1)
            logps_buf_p1[step].copy_(lp1)
            vals_buf_p1[step].copy_(v1.flatten())
            actions_buf_p2[step].copy_(a2)
            logps_buf_p2[step].copy_(lp2)
            vals_buf_p2[step].copy_(v2.flatten())

            # step env
            raw, reward, done, trunc, _ = envs.step({
                "player1": a1.cpu().numpy(),
                "player2": a2.cpu().numpy(),
                "belief": b.cpu().numpy()
            })  # add belief
            next_done = torch.tensor(np.logical_or(done, trunc), dtype=torch.float32, device=device)
            rews_buf_p1[step].copy_(torch.tensor(reward, device=device))
            rews_buf_p2[step].copy_(torch.tensor(-reward, device=device))

            # unpack next
            next_s1, next_t1, next_b, next_h1 = unpack(raw, "player1")
            next_s2, _, _, next_h2 = unpack(raw, "player2")



        # bootstrap values
        # od1 = {"state": next_s1, "belief": next_b, "player_type": next_t1, "action_history": next_h1}
        # od2 = {"state": next_s2, "belief": next_b, "action_history": next_h2}
        if args.include_action_history:
            od1 = {"state": next_s1, "player_type": next_t1, "belief": next_b, "action_history": next_h1}
            od2 = {"state": next_s2, "belief": next_b, "action_history": next_h2}
        else:
            od1 = {"state": next_s1, "belief": next_b, "player_type": next_t1}
            od2 = {"state": next_s2, "belief": next_b}

        with torch.no_grad():
            _, _, _, nv1, _ = agent1.model.get_action(od1)
            _, _, _, nv2, _ = agent2.model.get_action(od2)

        # compute GAE & returns
        adv1 = torch.zeros_like(rews_buf_p1)
        ret1 = torch.zeros_like(rews_buf_p1)
        adv2 = torch.zeros_like(rews_buf_p2)
        ret2 = torch.zeros_like(rews_buf_p2)
        last1 = last2 = 0
        for t in reversed(range(args.num_steps)):
            nonterm = 1.0 - (next_done if t == args.num_steps - 1 else dones_buf_p1[t+1])
            v_tp1 = nv1 if t == args.num_steps-1 else vals_buf_p1[t+1]
            v_tp2 = nv2 if t == args.num_steps-1 else vals_buf_p2[t+1]
            delta1 = rews_buf_p1[t] + args.gamma * v_tp1 * nonterm - vals_buf_p1[t]
            last1 = delta1 + args.gamma * args.gae_lambda * nonterm * last1
            adv1[t] = last1
            ret1[t] = last1 + vals_buf_p1[t]
            delta2 = rews_buf_p2[t] + args.gamma * v_tp2 * nonterm - vals_buf_p2[t]
            last2 = delta2 + args.gamma * args.gae_lambda * nonterm * last2
            adv2[t] = last2
            ret2[t] = last2 + vals_buf_p2[t]

        # flatten batches
        b_s1 = obs_state_buf_p1.reshape(-1, state_dim)
        b_t1 = obs_type_buf_p1.reshape(-1, num_type_p1)
        b_b  = obs_belief_buf.reshape(-1, )
        b_s2 = obs_state_buf_p2.reshape(-1, state_dim)
        b_a1 = actions_buf_p1.reshape(-1, action_dim)
        b_lp1= logps_buf_p1.reshape(-1)
        b_ret1= ret1.reshape(-1)
        b_adv1= adv1.reshape(-1)
        b_v1 = vals_buf_p1.reshape(-1)
        b_a2 = actions_buf_p2.reshape(-1, action_dim)
        b_lp2= logps_buf_p2.reshape(-1)
        b_ret2= ret2.reshape(-1)
        b_adv2= adv2.reshape(-1)
        b_v2 = vals_buf_p2.reshape(-1)

        if args.include_action_history:
            b_h1 = obs_hist_buf_p1.reshape(-1, action_hist_dim)
            b_h2 = obs_hist_buf_p2.reshape(-1, action_hist_dim)

        if args.norm_adv:
            b_adv1 = (b_adv1 - b_adv1.mean())/(b_adv1.std()+1e-8)
            b_adv2 = (b_adv2 - b_adv2.mean())/(b_adv2.std()+1e-8)

        # PPO update
        inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = inds[start:start+args.minibatch_size]

                if args.include_action_history:
                    mb_od1 = {
                        "state": b_s1[mb],
                        "player_type": b_t1[mb],
                        "belief": b_b[mb],
                        "action_history": b_h1[mb]
                    }
                    mb_od2 = {
                        "state": b_s2[mb],
                        "belief": b_b[mb],
                        "action_history": b_h2[mb]
                    }
                else:
                    mb_od1 = {
                        "state": b_s1[mb],
                        "player_type": b_t1[mb],
                        "belief": b_b[mb],
                    }
                    mb_od2 = {
                        "state": b_s2[mb],
                        "belief": b_b[mb],
                    }


                _, new_lp1, ent1, new_v1, _ = agent1.model.get_action(mb_od1, b_a1[mb])
                _, new_lp2, ent2, new_v2, _ = agent2.model.get_action(mb_od2, b_a2[mb])

                with torch.no_grad():
                    debug_obs = {
                        "state": torch.tensor([[0., -0.5, 0., 0., 0., 0.5, 0., 0., 0.]]),
                        "player_type": torch.tensor([[1., 0.]]),
                        "belief": torch.tensor([[0.5]]), }

                    means, _, _, _,  = agent1.model(debug_obs)


                new_v1 = new_v1.view(-1)
                new_v2 = new_v2.view(-1)
                r1 = (new_lp1 - b_lp1[mb]).exp()
                r2 = (new_lp2 - b_lp2[mb]).exp()

                # policy losses
                s1 = r1 * b_adv1[mb]
                s2 = torch.clamp(r1, 1-args.clip_coef, 1+args.clip_coef) * b_adv1[mb]
                pg1 = -torch.min(s1, s2).mean()
                s3 = r2 * b_adv2[mb]
                s4 = torch.clamp(r2, 1-args.clip_coef, 1+args.clip_coef) * b_adv2[mb]
                pg2 = -torch.min(s3, s4).mean()

                # value losses
                if args.clip_vloss:
                    v1_uc = (new_v1 - b_ret1[mb]).pow(2)
                    v1_c  = b_v1[mb] + torch.clamp(new_v1 - b_v1[mb], -args.clip_coef, args.clip_coef)
                    v1_cl = (v1_c - b_ret1[mb]).pow(2)
                    vloss1 = 0.5 * torch.max(v1_uc, v1_cl).mean()

                    v2_uc = (new_v2 - b_ret2[mb]).pow(2)
                    v2_c  = b_v2[mb] + torch.clamp(new_v2 - b_v2[mb], -args.clip_coef, args.clip_coef)
                    v2_cl = (v2_c - b_ret2[mb]).pow(2)
                    vloss2 = 0.5 * torch.max(v2_uc, v2_cl).mean()
                else:
                    vloss1 = 0.5 * (new_v1 - b_ret1[mb]).pow(2).mean()
                    vloss2 = 0.5 * (new_v2 - b_ret2[mb]).pow(2).mean()

                ent_loss1 = ent1.mean()
                ent_loss2 = ent2.mean()

                loss1 = pg1 - args.ent_coef*ent_loss1 + args.vf_coef*vloss1
                loss2 = pg2 - args.ent_coef*ent_loss2 + args.vf_coef*vloss2

                agent1.optimizer.zero_grad()
                loss1.backward()
                nn.utils.clip_grad_norm_(agent1.model.parameters(), args.max_grad_norm)
                agent1.optimizer.step()

                agent2.optimizer.zero_grad()
                loss2.backward()
                nn.utils.clip_grad_norm_(agent2.model.parameters(), args.max_grad_norm/10)
                agent2.optimizer.step()

        if iteration % args.print_freq == 0:
            print(f"Iter {iteration}/{args.num_iterations}, "
                  f"SPS {int(global_step/(time.time()-start_time))}, "
                  f"P1 V-loss {vloss1:.6f}, P1 PG-loss {pg1:.6f}\n"
                  f"P1 Means: {means[0, :, :]}")

        # for logging distance to gt
        with torch.no_grad():
            log_obs_1 = {
                "state": torch.tensor([[0., -0.5, 0., 0., 0., 0.5, 0., 0., 0.]]),
                "player_type": torch.tensor([[1., 0.]]),
                "belief": torch.tensor([[0.5]]), }

            log_obs_2 = {
                "state": torch.tensor([[0., -0.5, 0., 0., 0., 0.5, 0., 0., 0.]]),
                "player_type": torch.tensor([[0., 1.]]),
                "belief": torch.tensor([[0.5]]), }

            a1, _, _, _, _ = agent1.model.get_action(log_obs_1)
            a2, _, _, _, _ = agent1.model.get_action(log_obs_2)

            # for plotting policy viz
            log_obs_p2 = {
                "state": torch.tensor([[0., -0.5, 0., 0., 0., 0.5, 0., 0., 0.]]),
                "belief": torch.tensor([[0.5]]), }

            a_p2, _, _, _, _ = agent2.model.get_action(log_obs_p2)


            # gt action
            a1_gt = torch.tensor([[0.833, 1.818]])
            a2_gt = torch.tensor([[0.833, -1.818]])

            dist_to_gt_mean = (torch.linalg.norm(a1 - a1_gt) + torch.linalg.norm(a2 - a2_gt)) / 2

            log_data = {
                "iterations": iteration,
                "mean_dist_to_gt": dist_to_gt_mean.item(),
                "u1": means[0, 0, 1].item(), "u2": means[0, 1, 1].item(), "v1": a_p2[:, 1].item()
            }
            log_to_csv(log_data, args.save_dir + '/plot_logs.csv')

        # if iteration % args.save_interval == 0:
        #     torch.save(agent1.model.state_dict(),
        #                os.path.join(args.save_dir, f"agent1_{iteration}.pt"))
        #     torch.save(agent2.model.state_dict(),
        #                os.path.join(args.save_dir, f"agent2_{iteration}.pt"))

    envs.close()
    # torch.save(agent1.model.state_dict(), os.path.join(args.save_dir, "agent1_final.pt"))
    # torch.save(agent2.model.state_dict(), os.path.join(args.save_dir, "agent2_final.pt"))
