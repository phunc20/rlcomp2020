import os
import sys
import numpy as np
from game_socket_dummy import GameSocket
from miner_state import State
import os
import sys
#sys.path.append(os.path.abspath(os.path.pardir))
import constants02
from constants02 import terrain_ids, width, height, forest_energy



class MinerEnv:
    def __init__(self, host="localhost", port=1111):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score
        self.n_mines_visited = 0

    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
            self.n_mines_visited = 0
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_9x21x2_state(self):
        """
        Fuse `view` and `energyOnMap` into a single matrix to have a simple and concise state/observation.

        We want a matrix showing the following:
        `gold`: The amount of gold
        `all the others`: The energy that each type of terrain is going to take if being stepped into, e.g.
                          `land` => -1, `trap` => -10, etc.
        """
        view = (-9999) * np.ones((height, width), dtype=np.int32)
        for x in range(width):
            for y in range(height):
                if self.state.mapInfo.gold_amount(x, y) > 0:
                    view[y, x] = self.state.mapInfo.gold_amount(x, y)
        energyOnMap = np.array(self.socket.energyOnMap)

        ## `view` will contribute only to the type of terrain of `gold`
        # `energyOnMap` will contribute to the types of terrain of `land`, `trap`, `forest` and `swamp`.
        # Recall. `forest` was designated by BTC to the value of 0, to mean random integer btw [5..20].
        energyOnMap[energyOnMap == 0] = -forest_energy
        channel0 = np.maximum(view, energyOnMap)
        # Finish channel 0
        # Channel 1 will contain the position of the agent
        channel1 = np.zeros_like(channel0)
        x_agent_out_of_map = self.state.x < 0 or self.state.x >= width
        y_agent_out_of_map = self.state.y < 0 or self.state.y >= height
        if x_agent_out_of_map or y_agent_out_of_map:
            pass
        else:
            channel1[self.state.y, self.state.x] = self.state.energy
        state = np.stack((channel0, channel1), axis=-1)
        #return state.astype(np.int32)
        #return state.astype(np.float32)
        return state

    def get_9x21x2_tf_agent_state(self):
        state = self.get_9x21x2_state()
        return state.astype(np.float32)

    def get_deprecated_reward_01(self):
        # Initialize reward
        reward = 0

        # 4 action: Don't dig any more
        #score_action = self.state.score - self.score_pre
        #self.score_pre = self.state.score
        #if score_action > 0:
        #    #reward += score_action*(100 - self.state.stepCount)
        #    reward += score_action


        #s = self.get_state()
        ##print(f"self.state.x, self.state.y = {self.state.x}, {self.state.y} ")
        #terrain_now = s[self.state.y, self.state.x, 0]
        gold_here = self.state.mapInfo.gold_amount(self.state.x, self.state.y)
        #print(f"(x, y) = ({self.state.x}, {self.state.y}), gold_here = {gold_here}, energy = {self.state.energy}")
        #if terrain_now > 0:
        if gold_here > 0:
            # i.e. when we (agent) are standing on gold, i.e. when we finally arrive at gold
            #reward += terrain_now
            reward += gold_here
            # Remove gold_here (to push agent to find the next gold)
            for g in self.socket.stepState.golds:
                    if g.posx == self.state.x and g.posy == self.state.y:
                        self.socket.stepState.golds.remove(g)
                        self.n_mines_visited += 1
        

        #if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
        #    reward -= TreeID
        #if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
        #    reward -= TrapID
        #if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
        #    reward -= SwampID
        #    if self.state.lastAction == 4:
        #        reward -= 40

        #if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
        if self.state.status == constants02.agent_state_str2id["out_of_MAP"]:
            #if self.state.stepCount < 50:
            #    reward += -5*(50 - self.state.stepCount)
            reward -= 2000
        else:
            try:
                s = self.get_9x21x2_state()
                #print(f"self.state.x, self.state.y = {self.state.x}, {self.state.y} ")
                terrain_now = s[self.state.y, self.state.x, 0]
                if terrain_now < 0 and self.state.lastAction != constants02.rest:
                    # This substract the same amount of reward as energy when the agent steps into terrain_now, except for gold
                    reward += terrain_now
            except Exception:
                pass

            
        #if self.state.status == State.STATUS_STOP_END_STEP:
        if self.state.status == constants02.agent_state_str2id["no_more_STEP"]:
            #reward += (self.state.score/total_gold) * 100
            pass
        #if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
        if self.state.status == constants02.agent_state_str2id["no_more_ENERGY"]:
            if self.state.lastAction != constants02.rest:
                reward -= 500
        
        #if self.state.status == State.STATUS_PLAYING:
        if self.state.status == constants02.agent_state_str2id["PLAYing"]:
            reward += 1
            # We punish surplus `rest`
            if self.state.energy_pre == constants02.max_energy and self.state.lastAction == constants02.rest:
                reward -= 50

        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING



