import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import collections
from dataclasses import dataclass
import pickle
import zlib
import time
from gym.envs.classic_control import rendering


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

viewer = rendering.SimpleImageViewer()


def preprocess_frame(frame):

    image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
    image_resize = tf.image.resize(image_crop, [84, 84])
    image_scaled = tf.divide(image_resize, 255)

    frame = image_scaled.numpy()[:, :, 0]

    return frame


@dataclass
class Experience:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    
    def __init__(self, max_len, compress=True):
        self.max_len = max_len
        self.buffer = []
        self.compress = compress
        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
            transition : tuple(state, action, reward, next_state, done)
        """
        exp = Experience(*transition)
        exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0
        try:
            self.buffer[self.count] = exp
        except IndexError:
            self.buffer.append(exp)
        self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.buffer)
        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)
        
        selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        
        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)
        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)
        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)
        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]).astype(np.float32)
        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return (states, actions, rewards, next_states, dones)


class QNetwork(tf.keras.Model):

    def __init__(self, actions_space):
        super(QNetwork, self).__init__()
        self.action_space = actions_space
        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.qvalues = kl.Dense(self.action_space,
                                kernel_initializer="he_normal")

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        qvalues = self.qvalues(x)
        return qvalues

    def sample_action(self, x, epsilon=None):
        if (epsilon is None) or (np.random.random() > epsilon):
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions.numpy()[0]
        else:
            selected_action = np.random.choice(self.action_space)
        return selected_action

    def sample_actions(self, x):
        qvalues = self(x)
        selected_actions = tf.cast(tf.argmax(qvalues, axis=1), tf.int32)
        return selected_actions, qvalues


class DQNAgent:
    def __init__(self):

        self.env_name = "BreakoutDeterministic-v4"
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon_scheduler = (
            lambda steps: max(1.0 - 0.9 * steps / 1000000, 0.1))
        self.action_space = gym.make(self.env_name).action_space.n
        self.qnet = QNetwork(self.action_space)
        self.target_qnet = QNetwork(self.action_space)
        self.replay_buffer = ReplayBuffer(max_len=1000000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, epsilon=0.01/self.batch_size)
        self.huber_loss = tf.keras.losses.Huber()
        self.total_steps = 0
    
    def learn(self):
        #: 5000ゲーム学習する
        for episode in range(5000):
            #: 環境のリセットによる初期状態の取得
            env = gym.make(self.env_name)
            #: ゲーム画面を二値化したりトリミングしたりする前処理
            frame = preprocess_frame(env.reset())
            #: DQNでは直近4フレームの集合をstateとする
            frames = collections.deque(
                [frame] * 4, maxlen=4)
            #: 
            total_reward = 0
            
            #: breakoutの場合は残機5
            lives = 5
            done = False
            while not done:
                #: ゲーム画面を表示
                if episode % 10 == 0:
                    rgb = env.render('rgb_array')
                    upscaled=repeat_upsample(rgb,4, 4)
                    viewer.imshow(upscaled)
                    time.sleep(0.01)

                #: 総ステップ数に応じて探索率εを決定
                self.total_steps += 1
                epsilon = self.epsilon_scheduler(self.total_steps)
                
                #: 状態sに基づいてアクション決定
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=epsilon)
                next_frame, reward, done, info = env.step(action)
                frames.append(preprocess_frame(next_frame))
                next_state = np.stack(frames, axis=2)[np.newaxis, ...]
                total_reward += reward

                #: ライフが減ったら経験上はゲーム終了扱いとする
                if info["lives"] != lives:
                    lives = info["lives"]
                    transition = (state, action, reward, next_state, True)
                else:
                    transition = (state, action, reward, next_state, done)
                
                #: replay_bufferに遷移情報を蓄積
                self.replay_buffer.push(transition)

                if len(self.replay_buffer) > 50000:
                    #: 4ステップごとにQネットワークを更新
                    if self.total_steps % 4 == 0:
                        loss = self.update_network()
                    #: 10000ステップごとにtarget-QネットワークをQネットワークと同期
                    if self.total_steps % 10000 == 0:
                        self.target_qnet.set_weights(self.qnet.get_weights())

            print('Episode %d finished\t Reward %f' % (episode, total_reward))


    def update_network(self):
        #: リプレイバッファからミニバッチを取得
        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(batch_size=32)
        
        #: reward clipping
        rewards = np.clip(rewards, -1, 1)
        
        #: TQ = r + γmaxQ(s', a')
        next_actions, next_qvalues = self.target_qnet.sample_actions(next_states)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)
        max_next_qvalues = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=1, keepdims=True)
        target_q = rewards + self.gamma * (1 - dones) * max_next_qvalues

        with tf.GradientTape() as tape:
            qvalues = self.qnet(states)
            actions_onehot = tf.one_hot(
                actions.flatten().astype(np.int32), self.action_space)
            q = tf.reduce_sum(
                qvalues * actions_onehot, axis=1, keepdims=True)
            loss = self.huber_loss(target_q, q)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))
        
        return loss

        
start = DQNAgent()
start.learn()
print("fin")