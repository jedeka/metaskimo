import argparse
import importlib

import matplotlib.pyplot as plt
import torch
import wandb

from simpl.alg.spirl import ConstrainedSAC, PriorResidualNormalMLPPolicy
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector
from simpl.nn import itemize
from simpl.rl import MLPQF

load = torch.load('../checkpoints/spirl_pretrained_maze.pt', map_location='cpu')
print(load)