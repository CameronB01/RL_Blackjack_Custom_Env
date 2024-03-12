import gymnasium as gym
from stable_baselines3 import PPO, A2C
import numpy as np
import os
from BlackjackEnv import BlackEnv

models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = BlackEnv()
env.reset()

model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)

t_steps = 50000
episodes = 50

for i in range(100):
    model.learn(total_timesteps=t_steps, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{t_steps * i}")
        
env.close()