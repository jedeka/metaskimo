import numpy as np
import torch.distributions as torch_dist

from rolf.networks.distributions import TanhNormal, mc_kl


def testlog(*args, above=False, below=False, newline_cnt=1):
    """custom logger helper function"""
    import os, datetime
    if above: print('\n'*newline_cnt); print('*'*30)
    print(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | {os.path.basename(__file__)}]", end=' ')
    for i, content in enumerate(args):
        if i < len(args)-1: print(content,end=' ')
        else: print(content)
    if below: print('\n'); print('*'*30); print('\n'*newline_cnt)

def clipped_kl(a, b, clip=20):

    # testlog(f'{a}, {b}', above=True, below=True)


    kls = torch_dist.kl_divergence(a, b)
    # kls = mc_kl(a, b)
    scales =  kls.detach().clamp(0, clip) / kls.detach()
    return kls*scales

def inverse_softplus(x):
    return float(np.log(np.exp(x) - 1))

def inverse_sigmoid(x):
    return float(-np.log(1/x - 1))
