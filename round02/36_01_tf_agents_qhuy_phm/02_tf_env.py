import numpy as np

from tf_agents.environments import py_environment as pyenv
from tf_agents.environments import utils
from tf_agents.specs import array_spec 
from tf_agents.trajectories import time_step
from tf_agents.environments.tf_py_environment import TFPyEnvironment

import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
import constants02
from constants02 import width, height, agent_state_id2str

from miner_env import MinerEnv

class TFAgentsMiner(pyenv.PyEnvironment):
    def __init__(self, host="localhost", port=1111, debug=False):
        super(TFAgentsMiner, self).__init__()

        self.env= MinerEnv(host, port)
        #self.env= MinerEnv()
        self.env.start()
        self.debug = debug
        
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
        #self._observation_spec = array_spec.BoundedArraySpec(shape=(width*5,height*5,6), dtype=np.float32, name='observation')
        #self._observation_spec = array_spec.BoundedArraySpec(shape=(height, width, 2), dtype=np.float32, name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(height, width, 2), dtype=np.int32, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        mapID = np.random.randint(1, 6)
        #mapID = np.random.randint(0, 5)
        posID_x = np.random.randint(width)
        posID_y = np.random.randint(height)
        #request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        request = "map{},{},{},{},{}".format(mapID, posID_x, posID_y, constants02.max_energy, constants02.n_allowed_steps)
        self.env.send_map_info(request)
        self.env.reset()
        #observation = self.env.get_state()
        observation = self.env.get_9x21x2_tf_agent_state()

        return time_step.restart(observation)

    def _log_info(self):
        socket = self.env.socket

        # print(f'Map size:{self.socket.user.max_x, self.env.state.mapInfo.max_y}')
        #print(f"Self   - Pos ({socket.user.posx}, {socket.user.posy}) - Energy {socket.user.energy} - Status {socket.user.status}")
        print(f"Self   - Pos ({socket.user.posx}, {socket.user.posy}) - Energy {socket.user.energy}  ({agent_state_id2str[socket.user.status]})")
        for bot in socket.bots:
            print(f"Enemy  - Pos ({bot.info.posx}, {bot.info.posy}) - Energy {bot.info.energy}  ({agent_state_id2str[bot.info.status]})")
                
    def _step(self, action):
        if self.debug:
            self._log_info()
            
        self.env.step(str(action))
        #observation = self.env.get_state()
        observation = self.env.get_9x21x2_tf_agent_state()
        #reward = self.env.get_reward()
        reward = self.env.get_deprecated_reward_01()

        if not self.env.check_terminate():
            return time_step.transition(observation, reward)
        else:
            self.reset()
            return time_step.termination(observation, reward)

    def render(self):
        pass

if __name__ == '__main__':
    #env = TFAgentsMiner("localhost", 1111)
    env = TFAgentsMiner(debug=True)
    utils.validate_py_environment(env, episodes=5)
    tf_env = TFPyEnvironment(env)
