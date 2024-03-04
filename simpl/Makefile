a:
	@echo "type either 'make maze', 'make kitchen' or other commands."

m:
	@make maze

# usage: simpl_meta_train.py [-h] -g GPU -w WORKER_GPUS [WORKER_GPUS ...] -s SPIRL_PRETRAINED_PATH
#    [-t POLICY_VIS_PERIOD] [-p WANDB_PROJECT_NAME] [-r WANDB_RUN_NAME]
#    [-a SAVE_FILE_PATH]
#    {maze_40t,maze_20t,kitchen}

maze:
	python reproduce/simpl_meta_train.py maze_40t -s checkpoints/spirl_pretrained_maze.pt -g 0 -w 0 0 0 -p simpl@jamu -a ./checkpoints/meta_train.pt

fine_tune:
	python reproduce/simpl_fine_tune.py maze -g 0 -m ./checkpoints/meta_train.pt -s ./checkpoints/spirl_pretrained_maze.pt -p metaskimo_normal@jamu -mo checkpoints/normal_maze_pretrained.pt -sc config_maze.yaml

meta_train:
	python reproduce/simpl_meta_train.py maze_20t -s checkpoints/spirl_pretrained_maze.pt -g 0 -w 0 0 0 -p metaskimo@jamu -a ./checkpoints/meta_train.pt -m checkpoints/skimo_pretrained_maze.pt 
