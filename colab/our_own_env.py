#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

#tf.compat.v1.enable_v2_behavior()

import constants


class GoldMinerEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=5, name="action")
        # TODO: Find a good observation for this environment
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.int32, minimum=-3, name='observation')
        # Recall that _state reflects the progression of the game: game-start(0), in-progress(1) or game-over(2). No, bullshit. _state is the sum of cards on the agent's hands.
        self._state = 0
        self._energy = constants.max_energy
        self._episode_ended = False
        self.bots = [Bot1(), Bot2(), Bot3()]


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # TODO
        if action == constants.up:
            self._episode_ended = True
        elif action == constants.down:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        elif action == constants.left:
            pass
        elif action == constants.right:
            pass
        elif action == constants.rest:
            pass
        elif action == constants.dig:
            pass
        else:
            raise ValueError('`action` should be aomong 0..5.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                    np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

