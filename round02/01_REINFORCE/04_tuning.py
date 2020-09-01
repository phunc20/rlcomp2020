# In[2]:


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
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function, Checkpointer
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

import logging
import datetime

import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
import constants02
from constants02 import width, height, agent_state_id2str

from miner_env import MinerEnv
from tf_agents_miner_env import TFAgentsMiner


# In[3]:


n_iterations = 100_000
collect_episodes_per_iteration = 5
replay_buffer_capacity = 2000

lr_optimizer = 2.5e-4
log_interval = 25
n_eval_episodes = 20
eval_interval = 100
n_iterations_skipped = 5000

# In[5]:


train_env = TFPyEnvironment(TFAgentsMiner())
eval_env = TFPyEnvironment(TFAgentsMiner())


# In[6]:


conv_layer_params = [(4,(3,3),1), (8,(3,3),1)]
fc_layer_params = [128, 64, 32]
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
)


# In[7]:


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_optimizer)
#optimizer = keras.optimizers.RMSprop(lr=lr_optimizer, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)

#train_step_counter = tf.compat.v2.Variable(0)
global_step = tf.compat.v1.train.get_or_create_global_step()

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    #train_step_counter=train_step_counter,
    train_step_counter=global_step,
)

tf_agent.initialize()


# In[9]:


eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


# In[10]:


# **(?1)** Do we need to set `train_env.batch_size` in our case for the `miner_env`?

# In[11]:


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


# In[12]:


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity,
)


# In[14]:


def collect_episode(environment, policy, n_episodes):
    episode_counter = 0
    environment.reset()

    while episode_counter < n_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
#tf_agent.train_step_counter.assign(0)  # Can we get rid of this?
#global_step.assign(0)


#now = datetime.datetime.now()
#now_str = now.strftime("%Y%m%d-%H%M")
#script_name = __file__.split('.')[0]
#script_name = "03_checkpointer"
script_name = "04_tuning"
save_dir = os.path.join("models", script_name)
os.makedirs(save_dir, exist_ok=True)
#logging.basicConfig(filename=os.path.join(save_path, f"log-{now_str}.txt"),level=logging.DEBUG)
checkpoint_dir = os.path.join(save_dir, 'checkpoint')

train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=500,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, n_eval_episodes)
#avg_return = tf_metrics.AverageReturnMetric(eval_env, tf_agent.policy, n_eval_episodes)
returns = [avg_return]

for _ in range(n_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    #dataset = replay_buffer.as_dataset(single_deterministic_pass=True)
    #iterator = iter(dataset)
    #experience, buffer_info = next(iterator)
    #experience = next(iterator)

    #experience = replay_buffer.as_dataset(single_deterministic_pass=True)
    
    experience = replay_buffer.gather_all()
    train_loss = tf_agent.train(experience)
    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()
    #print("step = {}".format(step))

    #if step % log_interval == 0:
    #    print('step = {0:5d}  loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, n_eval_episodes)
        #avg_return = tf_metrics.AverageReturnMetric(eval_env, tf_agent.policy, n_eval_episodes)
        print('step = {0:5d}  Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        if step > n_iterations_skipped:
            train_checkpointer.save(global_step)

