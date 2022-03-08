from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy
from gym.envs.classic_control import rendering

import numpy as np
import time

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


# Frame-stacking with 4 frames
env = make_atari_env('PongNoFrameskip-v4', num_env=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model = DQN(CnnPolicy, env, verbose=1, tensorboard_log="./pong_tensorboard/")
model.learn(total_timesteps=1000000)
model.save("model_pong")

obs = env.reset()
viewer = rendering.SimpleImageViewer()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb,4, 4)
    viewer.imshow(upscaled)
    time.sleep(0.1)