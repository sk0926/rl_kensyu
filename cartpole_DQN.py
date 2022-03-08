from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.deepq.policies import MlpPolicy
from gym.envs.classic_control import rendering

import numpy as np
import time
import gym


env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./cartpole_tensorboard/")
# model = ACER('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("model_cartpole")

obs = env.reset()
viewer = rendering.SimpleImageViewer()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    env.render()
    time.sleep(0.05)

    if done:
        break
