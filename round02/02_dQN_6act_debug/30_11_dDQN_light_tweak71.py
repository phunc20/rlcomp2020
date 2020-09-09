########################################
# Changes compared to 30_11_dDQN_light_tweak14.py
# 01. 
#   trained on its checkpoint
# 02.
#   punish:
#     dig on obstacle
#     surplus rest
#     leave gold when gold > 100
# 03. (not yet change discount_rate)
########################################


import sys
import numpy as np
#import pandas as pd
import datetime
import json
from array import *
import os
import math
from random import randrange
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

import tensorflow.keras as keras

#import tensorflow.compat.v1 as tf
#from tensorflow.compat.v1.keras import backend as K
#tf.disable_v2_behavior()
import tensorflow as tf
from tensorflow.keras import backend as K

import logging
logging.basicConfig(level=logging.DEBUG)
import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
import constants02
import non_RL_agent
import non_RL_agent02
import non_RL_agent03
import non_RL_agent04
import non_RL_agent05
import non_RL_agent06
from miner_env import MinerEnv

n_episodes = 500_000
#n_epsilon_decay = int(n_episodes*.6)
#n_epsilon_decay = int(n_episodes*.805)
#n_epsilon_decay = 10**6 / 0.99
n_epsilon_decay = int(n_episodes // 50)
n_episodes_buf_fill = 10_000
batch_size = 32
discount_rate = 0.95
lr_optimizer = 2.5e-4
#lr_optimizer = 7.3e-4
#loss_fn = keras.losses.mean_squared_error
loss_fn = keras.losses.Huber()
max_replay_len = 50_000


Maps = [constants02.maps[i] for i in range(1, 6)]
env = MinerEnv() # Creating a communication environment between the DQN model and the game environment
env.start() # Connect to the game





from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


tf.random.set_seed(42)
np.random.seed(42)

#input_shape = [constants02.height, constants02.width, 1+4]
input_shape = [constants02.height, constants02.width, 1+1]
n_outputs = 6

model = keras.models.Sequential([
    Conv2D(4, 3, activation="relu", padding="same", input_shape=input_shape),
    #MaxPooling2D(2),
    Conv2D(8, 3, activation="relu", padding="same"),
    #Conv2D(128, 3, activation="relu", padding="same"),
    #MaxPooling2D(2),
    Flatten(),
    #Dense(128, activation="elu"),
    Dense(128, activation="elu"),
    Dense(64, activation="elu"),
    Dense(32, activation="elu"),
    Dense(n_outputs)
])
#h5 = "models/30_11_dDQN_light_tweak14/avg-1785.00-episode-11155-30_11_dDQN_light_tweak14-gold-1800-step-100-20200827-0903.h5"
#model = keras.models.load_model(h5)
target = keras.models.clone_model(model)
target.set_weights(model.get_weights())



from collections import deque
replay_memory = deque(maxlen=max_replay_len)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def epsilon_greedy_policy(state, epsilon=0, n_actions=6):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        #pictorial = pictorial_state(state)
        #Q_values = model.predict(pictorial[np.newaxis])
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    logging.debug(f"pos=({env.state.x:2d},{env.state.y:2d}), terrain={state[...,0][env.state.y, env.state.x]}, action={constants02.action_id2str[action]}, energy={env.state.energy}, score={env.state.score}")
    #next_state, reward, done, info = env.step(action)
    env.step(str(action))
    #next_state = env.get_9x21x2_state()
    next_state = env.get_view_9x21x5()[...,:2]
    reward = env.get_reward_6act_21()
    done = env.check_terminate()
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done


#optimizer = keras.optimizers.Adam(lr=1e-3)
#optimizer = keras.optimizers.Adam(lr=2.5e-4)
optimizer = keras.optimizers.Adam(lr=lr_optimizer)

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    #pictorials = np.array([pictorial_state(s) for s in states])
    #next_pictorials = np.array([pictorial_state(next_s) for next_s in next_states])
    #next_Q_values = model.predict(next_pictorials)
    next_Q_values = model.predict(next_states)
    #max_next_Q_values = np.max(next_Q_values, axis=1)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    #target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values
    target_Q_values = rewards + (1 - dones) * discount_rate * next_best_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        #all_Q_values = model(pictorials)
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


np.random.seed(42)
tf.random.set_seed(42)


from constants02 import n_allowed_steps

now = datetime.datetime.now()
now_str = now.strftime("%Y%m%d-%H%M")
script_name = __file__.split('.')[0]
save_path = os.path.join("models", script_name)
os.makedirs(save_path, exist_ok=True)


scores = [] 
scores_avg = [] 
best_score = 0
k = 10
scores_k_most_recent = deque([0]*k, maxlen=k)
best_score_avg = 1400

with open(os.path.join(save_path, f"log-{now_str}.txt"), 'w') as log:
    for episode in range(n_episodes):
        #mapID = np.random.randint(0, 5)
        mapID = np.random.randint(1,6)
        posID_x = np.random.randint(constants02.width) 
        posID_y = np.random.randint(constants02.height)
        request = "map{},{},{},50,100".format(mapID, posID_x, posID_y)
        env.send_map_info(request)
        env.reset()
        #obs = env.get_9x21x2_state()
        obs = env.get_view_9x21x5()[...,:2]
        delimiter = "==================================================="
        logging.debug(f"\n{delimiter}\nmapID {mapID}, start (x,y) = ({posID_x}, {posID_y}) on terrain {obs[...,0][posID_y, posID_x]} \n{delimiter}")
        undiscounted_return = 0
        for step in range(n_allowed_steps):
            logging.debug(f"(step {step:3d})")
            epsilon = max(1 - episode / n_epsilon_decay, 0.01)
            obs, reward, done = play_one_step(env, obs, epsilon)
            undiscounted_return += reward
            if done:
                break
        score = env.state.score
        scores.append(score)
        scores_k_most_recent.append(score)
        #score_avg = np.mean(scores_k_most_recent)
        score_avg = round(np.mean(scores_k_most_recent), 1)
        scores_avg.append(score_avg)
        #if score > best_score:
        if score_avg > best_score_avg:
            #best_weights = model.get_weights()
            #best_score_avg = score_avg 
            best_score_avg = min(1850, best_score_avg + 50)
            #best_score = score
            #model.save(os.path.join(save_path, f"episode-{episode+1}-gold-{env.state.score}-avg-{score_avg:4.2f}-step-{step+1}-{now_str}.h5"))
            model.save(os.path.join(save_path, f"avg-{score_avg:07.2f}-episode-{episode+1}-{__file__.split('.')[0]}-gold-{env.state.score}-step-{step+1}-{now_str}.h5"))
    
        #message = "(Episode {: 5d}/{})   Gold {: 4d}  avg {: 8.1f}  undisc_return {: 6d}   step {: 3d}   eps: {:.2f}  ({})\n".format(episode+1, n_episodes, env.state.score, score_avg, undiscounted_return, step + 1, epsilon, constants02.agent_state_id2str[env.state.status])
        message = "(Episode {:6d}/{})   Gold {:4d}  undisc_return {:8.0f}   step {:3d}   eps: {:.2f}  (map {}: {})\n".format(episode+1, n_episodes, env.state.score, undiscounted_return, step + 1, epsilon, mapID, constants02.agent_state_id2str[env.state.status])

        print(message, end='')
        log.write(message)
    
        #if episode > 500:
        if episode > n_episodes_buf_fill:
            training_step(batch_size)
        if episode % n_episodes_buf_fill == 0:
            target.set_weights(model.get_weights())

#np.save(f"scores-{now_str}", np.array(scores))
#np.save(f"scores-N-scores_avg-{now_str}", np.array([scores, scores_avg]))
np.save(f"scores-N-scores_avg-{__file__.split('.')[0]}-{now_str}", np.array([scores, scores_avg]))
