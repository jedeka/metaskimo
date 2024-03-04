import argparse
import importlib

import matplotlib.pyplot as plt
import torch
import wandb

from simpl.alg.spirl import ConstrainedSAC, PriorResidualNormalMLPPolicy
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector
from simpl.nn import itemize
from simpl.alg.simpl import ConditionedPolicy, ConditionedQF

from simpl.alg.skimo import SkiMoMetaAgent
import yaml 
import gym


from rolf.utils import LinearDecay

from tqdm import tqdm, trange
import numpy as np

import os, json, datetime
create_folder = lambda x : os.mkdir(x) if not os.path.exists(x) else None

def testlog(*args, above=False, below=False, newline_cnt=1):
    """custom logger helper function"""
    import os, datetime
    if above: print('\n'*newline_cnt); print('*'*30)
    print(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | {os.path.basename(__file__)}]", end=' ')
    for i, content in enumerate(args):
        if i < len(args)-1: print(content,end=' ')
        else: print(content)
    if below: print('\n'); print('*'*30); print('\n'*newline_cnt)


def simpl_fine_tune_iter(collector, trainer, *, batch_size, reuse_rate):
    log = {}
    
    
    # collect
    with trainer.policy.expl():
        episode = collector.collect_episode(trainer.policy)
    # testlog(episode, '\n', dir(episode), above=True, below=True)
    high_episode = episode.as_high_episode()
    trainer.buffer.enqueue(high_episode)
    log['tr_return'] = sum(episode.rewards)

    if trainer.buffer.size < batch_size:
        return log

    # train
    # n_step = int(reuse_rate * len(high_episode) / batch_size)
    # print(n_step)
    n_step = 60
    # testlog(f'iasbereiyugfb\n({reuse_rate} * {len(high_episode)} / {batch_size}\n', above=True, below=True)
    
    for _ in range(max(n_step, 1)):
        stat = trainer.step(batch_size)

    testlog(stat, above=True, below=True)
    
    log.update(itemize(stat))

    return log


if __name__ == '__main__':
    import_pathes = {
        'maze': 'maze.simpl_fine_tune',
        'kitchen': 'kitchen.simpl_fine_tune',
    }
    
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=import_pathes.keys())
    parser.add_argument('-g', '--gpu', required=True, type=int)
    parser.add_argument('-m', '--simpl-metatrained-path', required=True)
    parser.add_argument('-s', '--spirl-pretrained-path', required=True)
    
    parser.add_argument('-t', '--policy-vis_period', type=int)
    parser.add_argument('-p', '--wandb-project-name')
    parser.add_argument('-r', '--wandb-run-name')
    parser.add_argument('-mo', '--skimo')
    parser.add_argument('-sc', '--skimo_config')
    args = parser.parse_args()

    module = importlib.import_module(import_pathes[args.domain])
    env, tasks, config, visualize_env = module.env, module.tasks, module.config, module.visualize_env

    gpu = args.gpu
    simpl_metatrained_path = args.simpl_metatrained_path
    spirl_pretrained_path = args.spirl_pretrained_path
    policy_vis_period = args.policy_vis_period or 20
    wandb_project_name = args.wandb_project_name or 'SiMPL'
    wandb_run_name = args.wandb_run_name or args.domain + '.simpl_fine_tune.' + wandb.util.generate_id()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    seeds = [0, 1, 2]

    for seed in seeds:
        np.random.seed(seed)
        
        # load pre-trained SPiRL -> skill policy & prior
        load = torch.load(spirl_pretrained_path, map_location='cpu')
        horizon = load['horizon']
        high_action_dim = load['z_dim']
        spirl_low_policy = load['spirl_low_policy'].to(gpu).eval().requires_grad_(False)
        spirl_prior_policy = load['spirl_prior_policy'].to(gpu).eval().requires_grad_(False)

        # load meta-trained SiMPL -> task policy & encoder
        load = torch.load(simpl_metatrained_path, map_location='cpu')
        simpl_encoder = load['encoder'].to(gpu)
        simpl_high_policy = load['high_policy'].to(gpu)
        simpl_qfs = [qf.to(gpu) for qf in load['qfs']]
        simpl_alpha = load['policy_post_reg']

        testlog(f'\n{simpl_encoder}\n>>>\n{simpl_high_policy}\n', above=True, below=True)


        # SkiMo 
        skimo_pretrained_path = args.skimo
        skimo_config_path = args.skimo_config
        # load skimo config 
        from omegaconf import OmegaConf
        with open(skimo_config_path, 'r') as f:
            d = yaml.safe_load(f)
        cfg = OmegaConf.create(d)
        cfg_rolf = cfg.rolf
        cfg_rolf.pretrain_ckpt_path = skimo_pretrained_path
        meta_ac_space = gym.spaces.Box(-float('inf'), float('inf'), [cfg_rolf.skill_dim])
        ob_space = gym.spaces.Dict({"ob": gym.spaces.Box(-float('inf'), float('inf'), [state_dim])})

        # load skill dynamics, state encoder & decoder
        load = torch.load(skimo_pretrained_path, map_location='cpu')
        skimo_task_policy = SkiMoMetaAgent(cfg_rolf, ob_space, meta_ac_space)
        ckpt = load["agent"]
        skimo_task_policy.load_state_dict(ckpt["meta_agent"])
        
        # _horizon_decay = LinearDecay(1, cfg.n_skill, cfg.horizon_step)

        testlog('\n', 'dynamics\n', skimo_task_policy.model.dynamics,f"\n{'-'*20}\nstate encoder s -> h\n", skimo_task_policy.model.encoder, f"\n{'-'*20}\nstate decoder h -> s^\n", skimo_task_policy.decoder, above=True, below=True)


        # collector
        spirl_low_policy.explore = False
        collector = LowFixedHierarchicalTimeLimitCollector(env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit'])

        # for logging json
        dirname = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
        create_folder('json_logs'); create_folder(f'json_logs/{dirname}')


        # train on all tasks
        for task_idx, task in enumerate(tqdm(tasks)):
            wandb.init(
                project=wandb_project_name, name=wandb_run_name,
                config={
                    **config, 'task_idx': task_idx,
                    'spirl_pretrained_path': args.spirl_pretrained_path,
                    'simpl_metatrained_path': args.simpl_metatrained_path
                },
                mode='disabled',
            )
            
            logs = []

            with env.set_task(task):
                # collect from prior policy & encode
                prior_episodes = []
                for _ in range(config['n_prior_episode']):
                    e = simpl_encoder.encode([], sample=True)
                    with simpl_high_policy.expl(), simpl_high_policy.condition(e):
                        episode = collector.collect_episode(simpl_high_policy)
                    prior_episodes.append(episode)
                e = simpl_encoder.encode([episode.as_high_episode().as_batch() for episode in prior_episodes], sample=False)

                # ready networks
                high_policy = ConditionedPolicy(simpl_high_policy, e) # skill distribution conditioned for particular task
                qfs = [ConditionedQF(qf, e) for qf in simpl_qfs]
                buffer = Buffer(state_dim, high_action_dim, config['buffer_size'])
                # high policy
                trainer = ConstrainedSAC(high_policy, spirl_prior_policy, qfs, buffer,  
                                        skimo=skimo_task_policy,
                                        init_alpha=simpl_alpha, **config['constrained_sac']).to(gpu)

                # for episode_i in trange(config['n_prior_episode']+1, config['n_episode']+1):

                n_ep = 521 # TODO: hyperparam to be tuned, export later to config

                for episode_i in trange(config['n_prior_episode']+1, n_ep):
                    # testlog(config['train'])
                    log = simpl_fine_tune_iter(collector, trainer, **config['train'])

                    testlog('done iteration') #, above=True)

                    log['episode_i'] = episode_i

                    logs.append(log['tr_return'])

                    if episode_i % policy_vis_period == 0:
                        plt.close('all')
                        plt.figure()
                        log['policy_vis'] = visualize_env(plt.gca(), env, list(buffer.episodes)[-20:])
                    
                    testlog('episode done', below=True)
                    wandb.log(log)
                    
                
                
                
            with open(f'json_logs/{dirname}/seed{seed}_task{task_idx}.json', 'w+') as f:
                json.dump({task_idx : logs}, f)

            wandb.finish()

