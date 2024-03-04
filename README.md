# Skill-based Model-based Meta RL (MetaSkiMo) (Temporary)

## NOTE
More update TBD.

## Getting started
- We need to setup conda environment. For the experiment, we used conda with python==3.9
```
conda create -n metaskimo python=3.9
conda activate metaskimo
conda install mpi4py # optional, to prevent error during installation
```
- After that, install MuJoCo 2.1
* Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
* Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

- We basically install SkiMo and SiMPL. We need to first install all the dependencies from SkiMo and SiMPL, written inside the install.sh. You can try running:
```
sh install.sh
```
or 
```
#!/bin/bash

# skimo
pip install -r requirements.txt
pip install -e rolf/
pip install -e d4rl/
pip install -e spirl/

# simpl
pip install -r req_simpl.txt
pip install -e simpl/

```

## Running the code
Note: WandB can be used, however we set the default mode='disabled' for convenience purpose. If you want to use wandb, please configure the wandb in your computer, and set the mode to `enabled`
```
wandb.init(
    ...
    mode='disabled' # <-- change here 
)
```



1. First, we need to pretrain the skill dynamics 
```
python run.py --config-name skimo_maze run_prefix=test_pretrain gpu=0 wandb=false
```

2. Then, we need to run the meta training
```
cd simpl

python reproduce/simpl_meta_train.py maze_20t -s checkpoints/spirl_pretrained_maze.pt -g 0 -w 0 0 0 -p metaskimo@jamu -a ./checkpoints/meta_train.pt -m checkpoints/skimo_pretrained_maze.pt 

Note:
You can customize the saving name and the flags, such as the saving name in -a flag

```


3. Then, we can fine tune 
```
python reproduce/simpl_fine_tune.py maze -g 0 -m ./checkpoints/meta_train.pt -s ./checkpoints/spirl_pretrained_maze.pt -p metaskimo_normal@jamu -mo checkpoints/normal_maze_pretrained.pt -sc config_maze.yaml
```

For convenience, we provided the checkpoints.

For plotting, we use weighted moving average, similar to `plot.py`. 

## Troubleshooting 
- If there is the cython compiling problem related to `nogil`, refer to this link https://github.com/openai/mujoco-py/issues/773#issuecomment-1712434247. No.1 is the solution.
  * It might output the path for the .pyx file, such as `Cython.Compiler.Errors.CompileError: /home/<USER>/anaconda3/envs/<ENV NAME>/lib/python3.9/site-packages/mujoco_py/cymj.pyx`. Copy the path directory of the .pyx file, and open the builder.py, e.g. `/home/<USER>/anaconda3/envs/<ENV NAME>/lib/python3.9/site-packages/mujoco_py/builder.py`. 
  * Follow no.1 solution (adding the compiler directives) written in the mujoco-py issue link above

- If there is problem indicating package is not installed, install it by using `sudo apt install <package>`. e.g., `sudo apt install patchelf`

- If issue related to mpi4py installation during sh install.sh, install by using conda: `conda install mpi4py`
- For more troubleshooting, please read the link below:
  - https://github.com/clvrai/skimo
  - https://github.com/namsan96/SiMPL

