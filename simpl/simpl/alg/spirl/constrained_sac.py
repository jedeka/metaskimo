import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F

from simpl.math import clipped_kl, inverse_softplus
from simpl.nn import ToDeviceMixin

from collections import OrderedDict
# from torch.distributions import Normal, TransformedDistribution, Independent
# from torch.distributions.transforms import TanhTransform

from rolf.networks.distributions import TanhNormal, TanhTransform

def testlog(*args, above=False, below=False, newline_cnt=1):
    """custom logger helper function"""
    import os, datetime
    if above: print('\n'*newline_cnt); print('*'*30)
    print(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | {os.path.basename(__file__)}]", end=' ')
    for i, content in enumerate(args):
        if i < len(args)-1: print(content,end=' ')
        else: print(content)
    if below: print('\n'); print('*'*30); print('\n'*newline_cnt)

def force_exit(s='Exited manually'):
    raise SystemExit(s)

class ConstrainedSAC(ToDeviceMixin, nn.Module):
    def __init__(self, policy, prior_policy, qfs, buffer, skimo,
                 discount=0.99, tau=0.005, policy_lr=3e-4, qf_lr=3e-4,
                 auto_alpha=True, init_alpha=0.1, alpha_lr=3e-4, target_kl=1,
                 kl_clip=20, increasing_alpha=False):
        super().__init__()
        
        self.policy = policy
        self.prior_policy = prior_policy
        self.qfs = nn.ModuleList(qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in qfs])
        self.buffer = buffer

        self.discount = discount
        self.tau = tau

        self.policy_optim = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=qf_lr) for qf in qfs]

        self.auto_alpha = auto_alpha
        pre_init_alpha = inverse_softplus(init_alpha)
        if auto_alpha is True:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=alpha_lr)
            self.target_kl = target_kl
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.kl_clip = kl_clip
        self.increasing_alpha = increasing_alpha

        # skimo
        # with torch.no_grad():
        self.skimo_task_policy = skimo
        self.skill_dynamics = skimo.model.dynamics
        self.state_encoder = skimo.model.encoder
        self.state_decoder = skimo.decoder
        # testlog(dir(self.skimo_task_policy.model))
        self.skimo_task_policy.requires_grad_(False) 
        
    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
    
    def to(self, device):
        self.policy.to(device)
        return super().to(device)

    def step(self, batch_size):
        stat = {}
        batch = self.buffer.sample(batch_size).to(self.device)

        # qfs
        with torch.no_grad():
            target_qs = self.compute_target_q(batch)

        qf_losses = []
        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            qs = qf(batch.states, batch.actions)
            qf_loss = (qs - target_qs).pow(2).mean()

            qf_optim.zero_grad()
            qf_loss.backward()
            qf_optim.step()

            qf_losses.append(qf_loss)
        self.update_target_qfs()
        
        stat['qf_loss'] = torch.stack(qf_losses).mean()

        # high policy 
        # dists = self.policy.dist(batch.states)
        # policy_actions = dists.rsample() # skill
        # testlog("kjasbnefgiulbeszrdg\n", batch.states.size(), above=True, below=True)
        #z
        # ========== CEM planning ==================
        cem_actions, cem_scores = [], []
        for i, batch_state in enumerate(batch.states):
            # batch_state = batch.states[0] # comment later if want to use for loop
            batch_state = batch_state.unsqueeze(0)

            # testlog(f'batch_state: {batch_state.shape}')
            # self.skimo_task_policy.plann(policy_actions, cond_policy=self.policy)
            cfg = self.skimo_task_policy._cfg
            horizon = int(self.skimo_task_policy._horizon_decay(self.skimo_task_policy._step))
            
            ob = OrderedDict([('ob', batch.states[0])])
            state = self.state_encoder(ob) #latent state h
            z = state.repeat(cfg.num_policy_traj, 1)
            # z = state
            # testlog(f'\nz', above=True, below=True)


            # ddists = self.policy.dist(batch.states) # simpl
            # ppolicy_actions = ddists.rsample() # simpl

            # zz = state.repeat(cfg.num_policy_traj, 1)

            # process per index in batch 
            # ddists = self.policy.dist(batch.states) # simpl

            ddists = self.policy.dist(batch_state) # simpl
            ppolicy_actions = ddists.rsample() # simpl
            pp = ppolicy_actions.repeat(cfg.num_policy_traj, 1)
            # testlog(pp.shape, above=True)

            # testlog(TanhTransform(ddists, 1), above=True)
            # tanh_transform = TanhTransform(cache_size=1)
            # tanh_normal_dist = TransformedDistribution(ddists, [tanh_transform])
            # tanh_actions = tanh_normal_dist.rsample()
            # pp = tanh_actions.repeat(cfg.num_policy_traj, 1)

            # testlog(dir(ddists.base_dist), above=True)
            # tanh_normal_dist = TanhNormal(
            #     ddists.base_dist.loc.detach(),
            #     ddists.base_dist.scale.detach(),
            #     1
            # )
            # tanh_actions = tanh_normal_dist.rsample()
            # pp = tanh_actions.repeat(cfg.num_policy_traj, 1)
            # force_exit()

            # testlog(pp.shape, below=True)
            
            #clamp it later 
            
            # force_exit()

            policy_ac = []
            for t in range(horizon):
                # ddists = self.policy.dist(batch.states)
                # ppolicy_actions = ddists.rsample() # skill
                # policy_ac.append(self.actor.act(z))
                skimo_z = self.skimo_task_policy.actor.act(z)
                
                # ddists = self.policy.dist(batch.states) # simpl
                # ppolicy_actions = ddists.rsample() # simpl

                # testlog(self.state_encoder, above=True)
                # testlog(f'{skimo_z.size()}, {pp.size()}', below=True)
                # rsample, each ac could be different 
                # policy_ac.append(ppolicy_actions)
                policy_ac.append(pp)
                # policy_ac.append(skimo_z)
                z, _ = self.skimo_task_policy.model.imagine_step(z, policy_ac[t])
            policy_ac = torch.stack(policy_ac, dim=0)

            # testlog(f'zzzzx\n{z.shape}', above=True)

            # CEM optimization.
            z = state.repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)

            # testlog(f'zz after\n{z.shape}')

            mean = torch.zeros(horizon, self.skimo_task_policy._ac_dim, device=self.skimo_task_policy._device)
            std = 2.0 * torch.ones(horizon, self.skimo_task_policy._ac_dim, device=self.skimo_task_policy._device)
            # if prev_mean is not None and horizon > 1 and prev_mean.shape[0] == horizon:
            #     mean[:-1] = prev_mean[1:]
            
            
                
            for _ in range(cfg.cem_iter):
                # testlog(cfg.num_sample_traj + cfg.num_policy_traj)
                sample_ac = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                    horizon, cfg.num_sample_traj, self.skimo_task_policy._ac_dim, device=self.skimo_task_policy._device
                )
                sample_ac = torch.clamp(sample_ac, -0.999, 0.999)
                # sample_ac = sample_ac.repeat(cfg.num_policy_traj, 1)
                # testlog(f'sampleac\n{sample_ac.shape}') 
                # testlog(f'policyac\n{policy_ac.shape}', below=True)

                """
                [2023/12/23 17:55:36 | constrained_sac.py] sampleac
                torch.Size([1, 512, 10])
                [2023/12/23 17:55:36 | constrained_sac.py] policyac
                torch.Size([1, 6400, 10])
                
                # size : 1, 6400, 10 -> horizon is the first dimension. ac_dim: skill
                """
                
                ac = torch.cat([sample_ac, policy_ac], dim=1)
                
                imagine_return = self.skimo_task_policy.estimate_value(z, ac, horizon).squeeze(-1)
                _, idxs = imagine_return.sort(dim=0)

                idxs = idxs[-cfg.num_elites :]

                # testlog('pass idxs' , above=True)
                
                elite_value = imagine_return[idxs]
                elite_action = ac[:, idxs]
                
                # testlog('pass elite action')

                # Weighted aggregation of elite plans.
                score = torch.exp(cfg.cem_temperature * (elite_value - elite_value.max()))
                
                # testlog('pass2 here' , above=True,below=True)
                
                score = (score / score.sum()).view(1, -1, 1)
                new_mean = (score * elite_action).sum(dim=1)
                new_std = torch.sqrt(
                    torch.sum(score * (elite_action - new_mean.unsqueeze(1)) ** 2, dim=1)
                )

                mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * new_mean
                std = torch.clamp(new_std, self.skimo_task_policy._std_decay(self.skimo_task_policy._step), 2)

            # Sample action for MPC.
            score = score.squeeze().cpu().numpy()
            ac = elite_action[0, np.random.choice(np.arange(cfg.num_elites), p=score)]
            ac = torch.clamp(ac, -0.999, 0.999)
            
            cem_actions.append(ac)
            cem_scores.append(score)
        #=================================
        cem_actions = torch.stack(cem_actions, dim=0)

        # testlog(f'out from CEM!\nac shape: {ac.shape}', above=True, below=True)
        

        dists = self.policy.dist(batch.states)
        # policy_actions = dists.rsample() # skill
        
        # tanh_transform = TanhTransform(cache_size=1)
        # tanh_normal_dists = TransformedDistribution(dists, [tanh_transform])
        # tanh_actions = tanh_normal_dist.rsample()
        # pp = tanh_actions.repeat(cfg.num_policy_traj, 1)

        # tanh_normal_dists = TanhNormal(
        #         dists.base_dist.loc.detach(),
        #         dists.base_dist.scale.detach(),
        #         1
        #     )
        
        # testlog('v'*20, '\n', dists, above=True)
        # testlog(policy_actions.size())
        # testlog('^'*20, '\n', dists)
        # testlog(cem_actions.shape, below=True)

        # raise SystemExit("cut from inside ppolicy action calculation")
        policy_actions = cem_actions


        with torch.no_grad():
            prior_dists = self.prior_policy.dist(batch.states) # states_stack

        # testlog(tanh_normal_dists, prior_dists, above=True)
        # testlog(dir(tanh_normal_dists) , below=True)
        

        kl = torch_dist.kl_divergence(dists, prior_dists).mean(0)
        min_qs = torch.min(*[qf(batch.states, policy_actions) for qf in self.qfs])

        policy_loss = - min_qs.mean(0) + self.alpha * kl

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        stat['policy_loss'] = policy_loss
        stat['kl'] = kl
        stat['mean_policy_scale'] = dists.base_dist.scale.abs().mean()

        # alpha
        if self.auto_alpha is True:
            alpha_loss = (self.alpha * (self.target_kl - kl.detach())).mean()
            if self.increasing_alpha is True:
                alpha_loss = alpha_loss.clamp(-np.inf, 0)
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            stat['alpha_loss'] = alpha_loss
            stat['alpha'] = self.alpha

        # testlog('done step', below=True)
    
        return stat

    def compute_target_q(self, batch):
        dists = self.policy.dist(batch.next_states)
        actions = dists.sample()
        
        with torch.no_grad():
            prior_dists = self.prior_policy.dist(batch.next_states)
        kls = clipped_kl(dists, prior_dists, clip=self.kl_clip)
        min_qs = torch.min(*[target_qf(batch.next_states, actions) for target_qf in self.target_qfs])
        soft_qs = min_qs - self.alpha*kls
        return batch.rewards + (1 - batch.dones)*self.discount*soft_qs

    def update_target_qfs(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            for param, target_param in zip(qf.parameters(), target_qf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

