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

* XML files for Jaco robot with a ball and a block.
* Go to `/opt/anaconda3/envs/py35/lib/python3.5/site-packages/gym` to view and modify the xml/python files for some environment.
* Place `core.py` in the `gym` folder.
* Place `envs/__init__.py` in the `envs` folder.
* Place `envs/mujoco/assets/*` in the `envs/mujoco/assets` folder.
* Place `envs/mujoco/__init__.py,jaco.py,jaco_pick.py` in the `envs/mujoco` folder.
* To run the HER code on the Jaco environment, go to `baselines/baselines/her/her.py` and change `env.spec.id` to `env.unwrapped.spec.id`.
