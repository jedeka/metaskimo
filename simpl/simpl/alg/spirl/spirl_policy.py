import torch
import torch.distributions as torch_dist
import torch.nn as nn

from simpl.rl import StochasticNNPolicy, ContextPolicyMixin

def testlog(*args, above=False, below=False, newline_cnt=1):
    """custom logger helper function"""
    import os, datetime
    if above: print('\n'*newline_cnt); print('*'*30)
    print(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | {os.path.basename(__file__)}]", end=' ')
    for i, content in enumerate(args):
        if i < len(args)-1: print(content,end=' ')
        else: print(content)
    if below: print('\n'); print('*'*30); print('\n'*newline_cnt)



class SpirlMLP(nn.Module):
    def __init__(self, dims, activation='relu'):
        super().__init__()
        
        layers = [
            nn.Linear(dims[0], dims[1]), # 4, 128
            nn.LeakyReLU(0.2, inplace=True)
        ]
                            
        prev_dim = dims[1] # 128
        for dim in dims[2:-1]:
            layers.append(nn.Linear(prev_dim, dim, bias=False))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(prev_dim, dims[-1]))
        self.net = nn.Sequential(*layers)

        testlog('from spirlMLP class', dims, above=True, below=True)

    def forward(self, x):
        return self.net(x)
    
# SELF-NOTE: SpirlMLP must be replaced by skimo encoder, and actor(ActionDecoder)


class SpirlLowPolicy(ContextPolicyMixin, StochasticNNPolicy):
    def __init__(self, state_dim, z_dim, action_dim, hidden_dim, n_hidden, prior_state_dim=None):
        super().__init__()
        self.net = SpirlMLP([state_dim + z_dim] + [hidden_dim]*n_hidden + [action_dim])
        self.log_sigma = nn.Parameter(-50*torch.ones(action_dim))
        self.z_dim = z_dim
        self.prior_state_dim = prior_state_dim
                
    def dist(self, batch_state_z):
        if self.prior_state_dim is not None:
            batch_state_z = torch.cat([
                batch_state_z[..., :self.prior_state_dim],
                batch_state_z[..., -self.z_dim:]
            ], dim=-1)
        loc = self.net(batch_state_z)
        scale = self.log_sigma.clamp(-10, 2).exp()[None, :].expand(len(loc), -1)
        dist = torch_dist.Normal(loc, scale)
        return torch_dist.Independent(dist, 1)


class SpirlPriorPolicy(StochasticNNPolicy):    
    def __init__(self, state_dim, z_dim, hidden_dim, n_hidden, prior_state_dim=None):
        super().__init__()
        # dim for maze: [4] + [128]*n_hidden + [2*20]
        self.net = SpirlMLP([state_dim] + [hidden_dim]*n_hidden + [2*z_dim]) 
        self.prior_state_dim = prior_state_dim
        
        # self.state_dim = state_dim
        # self.z_dim = z_dim
        # self.hidden_dim = hidden_dim
        # self.n_hidden = n_hidden
        # self.prior_state_dim = prior_state_dim

        testlog("printing from SpirlPriorPolicy.init()", above=True)
        testlog([state_dim] + [hidden_dim]*n_hidden + [2*z_dim], below=True)
        

    def dist(self, batch_state):
        # prior_state_dim = 4 -> ob_space
        # testlog("PRINTING DIST INITIALIZATION PARAMS:", f'{[self.state_dim]} + {[self.hidden_dim]}*{self.n_hidden} + {[2*self.z_dim]}', below=True)
        
        if self.prior_state_dim is not None:
            batch_state = batch_state[..., :self.prior_state_dim]
        loc, log_scale = self.net(batch_state).chunk(2, dim=-1)
        # testlog("printing from SpirlPriorPolicy.dist()", above=True)
        # testlog(self.net, f'\n\n\n{self.net(batch_state)},\nchunk:\n{self.net(batch_state).chunk(2,dim=1)}\n\n\n prior_state_dim: {self.prior_state_dim}', f'loc, log_scale: {loc.size(), log_scale.size()}', below=True)
        # size of loc and log_scale: (torch.Size([30720, 10]), torch.Size([30720, 10])
        
        # Q: is 10 the number of skills?
        # Linear(in_features=128, out_features=20, bias=True) -> Q: is 20 = 2 (mean, log_scale) * 10
        
        dist = torch_dist.Normal(loc, log_scale.clamp(-10, 2).exp())
        return torch_dist.Independent(dist, 1)
    
    def dist_param(self, batch_state):
        if self.prior_state_dim is not None:
            batch_state = batch_state[..., :self.prior_state_dim]
        loc, log_scale = self.net(batch_state).chunk(2, dim=-1)
        return loc, log_scale.clamp(-10, 2)
