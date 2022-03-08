# coding:utf-8
# [0]ライブラリのインポート
import gym  #倒立振子(cartpole)の実行環境
import numpy as np
import time
import os


# [1]Q関数を離散化して定義する関数　------------
# 観測した状態を離散値にデジタル変換する
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 各値を離散値に変換
def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        #np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        #np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_split1)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_split2))
    ]
    return sum([x * (num_split1**i) for i, x in enumerate(digitized)])


# [2]行動a(t)を求める関数 -------------------------------------
def get_action(next_state, episode):
    #ε-greedy法
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action


# [3]Qテーブルを更新する関数 -------------------------------------
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q=max(q_table[next_state][0],q_table[next_state][1] )
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * next_Max_Q)

    return q_table


# [4]. メイン関数開始 パラメータ設定--------------------------------------------------------
env = gym.make('CartPole-v0')
max_number_of_steps = 1000  #1試行のstep数
num_episodes = 1000  #総試行回数
num_render = 0  #表示開始の試行回数
# 状態を分割してQ関数（表）を作成
num_split1 = 8  #分割数1
num_split2 = 6  #分割数2
q_table = np.random.uniform(
    low=-1, high=1, size=(num_split1*num_split2, env.action_space.n))
np.set_printoptions(precision=1, suppress=True)  #print用フォーマット


# [5] メインルーチン--------------------------------------------------
for episode in range(num_episodes):  #試行数分繰り返す
    # 環境の初期化
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):  #1試行のループ
        if episode > num_render:
            # cartPoleを描画する
            env.render()
            time.sleep(0.001)

        # 行動a_tの実行により、s_{t+1}, r_{t}などを計算する
        observation, reward, done, info = env.step(action)
        cart_pos, cart_v, pole_angle, pole_v = observation

        # 報酬を設定し与える
        if abs(pole_angle) > 0.8 or abs(cart_pos) > 2.4 or t > 200:
            if t < 200:
                reward = -200  #こけたら罰則
            else:
                reward = 1  #立ったまま終了時は罰則はなし

            episode_reward += reward
            print("Episode ", episode, "\treward", episode_reward)
            break
        else:
            reward = 1  #各ステップで立ってたら報酬追加

        episode_reward += reward  #報酬を追加

        # 離散状態s_{t+1}を求め、Q関数を更新する
        next_state = digitize_state(observation)  #t+1での観測状態を、離散値に変換
        q_table = update_Qtable(q_table, state, action, reward, next_state)   

        #  次の行動a_{t+1}を求める 
        action = get_action(next_state, episode)    # a_{t+1} 

        state = next_state

        # if episode > num_render:
            # ここにq_tableを表示
            # q_table_print = np.reshape(q_table, (1,num_split1*num_split2*2),'F')
            # os.system('cls')
            # print(np.reshape(q_table_print, (2,num_split2,num_split1)))
