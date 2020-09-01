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


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


if __name__ == '__main__':
    #env = TFAgentsMiner("localhost", 1111)
    #env = TFAgentsMiner(debug=True)
    env = TFAgentsMiner()
    #utils.validate_py_environment(env, episodes=5)
    tf_env = TFPyEnvironment(env)

    ## Creating the Deep Q-Network
    preprocessing_layer = keras.layers.Lambda(
        lambda obs: tf.cast(obs, np.float32)/255.)
    conv_layer_params = [(4,(3,3),1), (8,(3,3),1)]
    fc_layer_params = [128, 64, 32]

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )

    ## Creating the DQN Agent
    train_step = tf.Variable(0)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    #update_period = 4 # run a training step every 4 collect steps
    update_period = 100
    #optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0, epsilon=0.00001, centered=True)
    optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        #decay_steps=3_800_000,
        decay_steps=380_000,
        end_learning_rate=0.01) # final ε
    agent = DqnAgent(tf_env.time_step_spec(),
                     tf_env.action_spec(),
                     q_network=q_net,
                     optimizer=optimizer,
                     target_update_period=2000,
                     td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                     gamma=0.99, # discount factor
                     #train_step_counter=train_step,
                     train_step_counter=global_step,
                     epsilon_greedy=lambda: epsilon_fn(train_step),
    )
    agent.initialize()
    #print(f"tf_env.batch_size = {tf_env.batch_size}")


    ## Replay buffer and observer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1_000_000)
    
    replay_buffer_observer = replay_buffer.add_batch

    ## Training metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]
    logging.getLogger().setLevel(logging.INFO)
    log_metrics(train_metrics)

    ## Creating the collect driver
    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=update_period,
    )


    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                            tf_env.action_spec())
    init_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(10000)],
        num_steps=10000)
    final_time_step, final_policy_state = init_driver.run()


    ## Create dataset
    dataset = replay_buffer.as_dataset(
        sample_batch_size=64,
        num_steps=2,
        num_parallel_calls=3).prefetch(3)


    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

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

    def train_agent(n_iterations):
        time_step = None
        policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
        iterator = iter(dataset)
        for iteration in range(n_iterations):
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            print("\r{} loss:{:.5f}".format(
                iteration, train_loss.loss.numpy()), end="")
            if iteration % 100 == 0:
                log_metrics(train_metrics)
            # TODO: save checkpoints
            if iteration >= 10_000 and iteration % 100 == 0:
                train_checkpointer.save(global_step)
                #model.save(os.path.join(save_path, f"avgGold-{score_avg:06.1f}-iter-{iteration+1}-{__file__.split('.')[0]}-visitRatio-{aux_score}-gold-{env.state.score}-step-{step+1}-{now_str}.h5"))

    train_agent(n_iterations=500_000+1)
