from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.deepq.policies import MlpPolicy
from gym.envs.classic_control import rendering

import numpy as np
import time
import gym

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


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

    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb,4, 4)
    viewer.imshow(upscaled)
    time.sleep(0.01)
