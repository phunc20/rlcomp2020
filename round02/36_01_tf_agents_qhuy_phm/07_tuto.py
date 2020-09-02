n_iterations = 1_000
n_eval_episodes = 20
#collect_episodes_per_iteration = 5
replay_buffer_capacity = 20

batch_size = 64
collect_steps_per_iteration = 1
lr_optimizer = 2.5e-4
log_interval = 5
#n_eval_episodes = 20
eval_interval = 10
#n_iterations_skipped = 5000

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

#from tf_agents.environments import py_environment as pyenv, tf_py_environment, utils
#from tf_agents.specs import array_spec 
from tf_agents.trajectories import time_step
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function, Checkpointer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver

import logging
import datetime

import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
import constants02
from constants02 import width, height, agent_state_id2str

from miner_env import MinerEnv
from tf_agents_miner_env import TFAgentsMiner


def compute_avg_return(environment, policy, n_episodes=10):
    total_return = 0.0
    for _ in range(n_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / n_episodes
    return avg_return.numpy()[0]



if __name__ == '__main__':
    train_env = TFPyEnvironment(TFAgentsMiner())
    eval_env = TFPyEnvironment(TFAgentsMiner())

    ## Creating the Deep Q-Network
    preprocessing_layer = keras.layers.Lambda(
        lambda obs: tf.cast(obs, np.float32)/255.)
    conv_layer_params = [(4,(3,3),1), (8,(3,3),1)]
    fc_layer_params = [128, 64, 32]

    q_net = QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )

    ## Creating the DQN Agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        decay_steps=380_000,
        end_learning_rate=0.01) # final ε
    agent = DqnAgent(train_env.time_step_spec(),
                     train_env.action_spec(),
                     q_network=q_net,
                     optimizer=optimizer,
                     target_update_period=50,
                     td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                     gamma=0.95, # discount factor
                     train_step_counter=global_step,
                     epsilon_greedy=lambda: epsilon_fn(global_step),
    )
    agent.initialize()
    #print(f"tf_env.batch_size = {tf_env.batch_size}")


    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    ## Replay buffer and observer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    def collect_step(environment, policy, buffer_):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
    
        # Add trajectory to the replay buffer
        buffer_.add_batch(traj)
    
    def collect_data(env, policy, buffer_, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer_)
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    collect_data(train_env, random_policy, replay_buffer, steps=5000)

    # Dataset generates trajectories with shape [B,2,...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2).prefetch(3)
    
    iterator = iter(dataset)

    
    agent.train = function(agent.train)
    agent.train_step_counter.assign(0)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M")
    script_name = __file__.split('.')[0]
    save_dir = os.path.join("models", script_name)
    os.makedirs(save_dir, exist_ok=True)
    #logging.basicConfig(filename=os.path.join(save_path, f"log-{now_str}.txt"),level=logging.DEBUG)
    checkpoint_dir = os.path.join(save_dir, 'checkpoint')
    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=500,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )



    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, n_eval_episodes)
    returns = [avg_return]
    
    for _ in range(n_iterations):
        # Collect a few episodes using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        #print(experience)
        #train_loss = agent.train(experience)
        train_loss = agent.train(experience).loss
        replay_buffer.clear()
    
        step = agent.train_step_counter.numpy()
        print("\rstep = {}".format(step), end='')
    
        if step % log_interval == 0:
            print('step = {0:5d}  loss = {1}'.format(step, train_loss.loss))
    
        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, n_eval_episodes)
            print('step = {0:5d}  Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            #if step > n_iterations_skipped:
            #    train_checkpointer.save(global_step)

