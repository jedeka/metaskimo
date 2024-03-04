all:
	@echo "open Makefile to see commands"



# skimo
# test pretrain
pretrain_maze:
	python run.py --config-name skimo_maze run_prefix=test_pretrain gpu=0 wandb=false

	
meta_train:
	python simpl/reproduce/simpl_meta_train.py maze_20t -s simpl/checkpoints/spirl_pretrained_maze.pt -g 0 -w 0 0 0 -p metaskimo -a simpl/checkpoints/meta_train.pt -m simpl/checkpoints/skimo_pretrained_maze.pt -sc simpl/config_maze.yaml

fine_tune:
	python simpl/reproduce/simpl_fine_tune.py maze -g 0 -m ./checkpoints/meta_train.pt -s ./checkpoints/spirl_pretrained_maze.pt -p metaskimo -mo checkpoints/normal_maze_pretrained.pt -sc config_maze.yaml
