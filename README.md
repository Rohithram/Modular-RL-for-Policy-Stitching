# Policy-Stitching

Modular Reinforcement Learning with policy stitching

## DDPG

* To run the DDPG algorithm, clone the [spinningup](https://github.com/openai/spinningup) repository of openai and `cd spinningup`.
* Copy the file `run_ddpg.py` into the folder and then execute `python run_ddpg.py` along with some optional arguments.

## HER

* Clone the [baselines](https://github.com/openai/baselines) repository and `cd baselines`.
* Execute `python -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=5000 --play --load_path=results`.
* In case, you encounter the error: `ERROR: GLEW initalization error: Missing GL version`, then add the following line to your `.bashrc` or `.zshrc` file:  
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`.

## Modified Models
XML files for Jaco robot with a ball and a block


### TO DO
* Jaco Python Environment 
* Run HER on Jaco env
* RL Algo implementation
