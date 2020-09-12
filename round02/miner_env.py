import os
import sys
import numpy as np
from game_socket_dummy import GameSocket
from miner_state import State
import os
import sys
import logging
#sys.path.append(os.path.abspath(os.path.pardir))
import constants02
#from constants02 import terrain_ids, width, height, forest_energy
from constants02 import *

class MinerEnv:
    def __init__(self, host="localhost", port=1111):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score
        self.n_mines_visited = 0
        self.view_9x21x5 = None
        self.view_9x21x5_prev = None  # previous frame
        #self.channel2 = None
        #self.channel2_prev = None
        self.start_tracking_swamp = False
        self.other_players = []
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                self.other_players.append((player["playerId"], player))
            else:
                self.player_main = player
        self.other_players = sorted(self.other_players)
        self.n_gold_4act_agent = 0
        self.n_gold_5act_agent = 0

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
            self.view_9x21x5 = None
            self.view_9x21x5_prev = None  # previous frame
            #self.channel2 = None
            #self.channel2_prev = None
            self.start_tracking_swamp = False
            self.other_players = []
            for player in self.state.players:
                if player["playerId"] != self.state.id:
                    self.other_players.append((player["playerId"], player))
            self.other_players = sorted(self.other_players)
            self.n_gold_4act_agent = 0
            self.n_gold_5act_agent = 0
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

    def get_view_9x21x5(self):
        """
        self.state.mapInfo.get_obstacle(x, y) returns an int
        and has the following meaning:
         0: land
         1: forest
         2: trap
         3: swamp
        -1: gold
        """
        #print(f"self.view_9x21x5 is None: {self.view_9x21x5 is None}")
        if not self.view_9x21x5 is None:
            #print(f"view_9x21x5 copied to view_9x21x5_prev")
            self.view_9x21x5_prev = self.view_9x21x5.copy()
        ## channel0 will contain the ids of each terrain
        ## channel1 will contain the value of each terrain, notably
        ##          it can track swamp's value
        ## channel2 agent's position
        ## channel3 Bot1's position
        ## channel4 Bot2's position
        ## channel5 Bot3's position
        channel0 = np.empty([height, width], dtype=np.float32)
        for x in range(width):
            for y in range(height):
                    obstacle_id = self.state.mapInfo.get_obstacle(x, y)
                    if obstacle_id == -1: # i.e. if (x, y) gold
                        channel0[y, x] = self.state.mapInfo.gold_amount(x, y)
                    else:
                        channel0[y, x] = obstacle_id
        #print(f"channel0 =\n{channel0.astype(np.int16)}")
        ## Let's construct channel2-5 first,
        ## since the construction of channel1 is much more involved.
        ## (Note. we'll temporarily use 1 to show bots' pos)
        channel2 = np.zeros([height, width], dtype=np.float32)
        try:
            #logging.debug(f"self.state.x, self.state.y = {self.state.x}, {self.state.y}")
            channel2[self.state.y, self.state.x] = self.state.energy
        except IndexError as e:
            # When agent out of MAP, we just leave channel2 zero everywhere
            pass

        channels_other_players = np.zeros([height, width, 3], dtype=np.float32)
        #channel3 = np.zeros([height, width], dtype=np.float32)
        #channel4 = np.zeros([height, width], dtype=np.float32)
        #channel5 = np.zeros([height, width], dtype=np.float32)
        #self.other_players = []
        index = 0
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                #self.other_players.append((player["playerId"], player))
                x = player["posx"]
                y = player["posy"]
                channels_other_players[..., index][y, x] = 1
                index += 1
        ##self.other_players = []
        ##for player in self.state.players:
        ##    if player["playerId"] != self.state.id:
        ##        self.other_players.append((player["playerId"], player))
        ##self.other_players = sorted(self.other_players)
        #for k, player in enumerate(self.other_players):
        #    # player[0] is its ID, player[1] its information.
        #    xk = player[1]["posx"]
        #    yk = player[1]["posy"]
        #    channels_other_players[..., k][yk, xk] = 1

        ## Construction of channel1
        ## depending on whether beginning of an episode (aka BOE)
        if not self.start_tracking_swamp:
            #print(f"self.start_tracking_swamp = {self.start_tracking_swamp}")
            self.start_tracking_swamp = True
            channel1 = channel0.copy()
            #channel1[channel1 == terrain_ids["land"]] = punishments["land"]
            channel1[channel0 == terrain_ids["land"]] = punishments["land"]
            # forest will be "overestimated" by its highest strike
            #channel1[channel1 == terrain_ids["forest"]] = punishments["forest"]
            channel1[channel0 == terrain_ids["forest"]] = punishments["forest"]
            
            #channel1[channel1 == terrain_ids["trap"]] = punishments["trap"]
            channel1[channel0 == terrain_ids["trap"]] = punishments["trap"]
            ## BOE => all swamp values are -5, but we'll make it -5.1
            #channel1[channel1 == terrain_ids["swamp"]] = punishments["swamp"]
            channel1[channel0 == terrain_ids["swamp"]] = punishments["swamp"]
            # gold will be kept unchanged
            #self.view_9x21x5 = np.stack((channel1, channel2, channel3, channel4, channel5), axis=-1)
            self.view_9x21x5 = np.stack((channel1, channel2, channels_other_players[..., 0], channels_other_players[..., 1], channels_other_players[..., 2]), axis=-1)
            #self.view_9x21x5 = np.stack((channel1, channel2), axis=-1)
            #print(f"(1st entering) view =\n{self.view_9x21x5[...,0].astype(np.int16)}\nview_pre =\n{self.view_9x21x5_prev}")
            return self.view_9x21x5
        else:
            #print(f"self.start_tracking_swamp = {self.start_tracking_swamp}")
            #print(f"(2nd entering) view_pre =\n{self.view_9x21x5_prev[...,0].astype(np.int16)}")
            # Update self.view_9x21x5
            self.view_9x21x5[...,1] = channel2
            for i in range(2, 5):
                self.view_9x21x5[...,i] = channels_other_players[...,i-2]

            #channel1 = self.view_9x21x5_prev[..., 1]
            channel1 = self.view_9x21x5_prev[..., 0]
            #print(f"channel1 =\n{channel1.astype(np.int16)}")
            ## land is permanent; but gold may become land
            channel1[channel0 == terrain_ids["land"]] = punishments["land"]
            ## forest is permanent
            #channel1[channel0 == terrain_ids["forest"]] = punishments["forest"]
            ## trap may become land if sb steps on it; but it only DECREASEs
            #channel1[channel0 == terrain_ids["trap"]] = punishments["trap"]
            ## Update gold's value
            #channel1[channel0 > 0] = channel0[channel0 > 0]
            channel1[channel0 > 3] = channel0[channel0 > 3]
            ## swamp is also permanent, but its value may change
            #for player in self.state.players:
            #    if is_a_mv(player["lastAction"]):
            #        x_now = player["posx"]
            #        y_now = player["posy"]
            #        is_swamp = channel0[y_now, x_now] == terrain_ids["swamp"]
            #        if is_swamp:
            #            channel1[y_now, x_now] = dict_bog[channel1[y_now, x_now]]
            for c in range(1, 5):
                moved = not np.array_equal(self.view_9x21x5[...,c] - self.view_9x21x5_prev[...,c], np.zeros_like(self.view_9x21x5[...,c]))
                if moved:
                    row, col = np.unravel_index(np.argmax(self.view_9x21x5[...,c], axis=None), self.view_9x21x5[...,c].shape)
                    is_swamp = channel0[row, col] == terrain_ids["swamp"]
                    if is_swamp:
                        #print(f"row, col = {row}, {col}")
                        #print(f"channel1[row, col] = {channel1[row, col]}")
                        ## N.B.
                        ## In [12]: np.array([-5.1],dtype=np.float32)[0] == -5.1
                        ## Out[12]: False
                        ## 
                        ## In [13]: np.array([-5.1],dtype=np.float64)[0] == -5.1
                        ## Out[13]: True
                        ## 
                        ## In [14]: np.array(-5.1,dtype=np.float64) == -5.1
                        ## Out[14]: True
                        ## 
                        ## In [15]: np.array(-5.1,dtype=np.float32) == -5.1
                        ## Out[15]: False
                        channel1[row, col] = dict_bog[channel1[row, col]]
                        #channel1[row, col] = dict_bog[np.round(channel1[row, col],1)]
            #self.view_9x21x5 = np.stack((channel1, channel2), axis=-1)
            self.view_9x21x5 = np.stack((channel1, channel2, channels_other_players[..., 0], channels_other_players[..., 1], channels_other_players[..., 2]), axis=-1)
            #print(f"view =\n{self.view_9x21x5[...,0].astype(np.int16)}")
            return self.view_9x21x5

    def get_198_state(self):
        view = np.zeros([self.state.mapInfo.max_y + 1, self.state.mapInfo.max_x + 1], dtype=int)
        for x in range(self.state.mapInfo.max_x + 1):
            for y in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(x, y) == TreeID:  # Tree
                    view[y, x] = -TreeID
                if self.state.mapInfo.get_obstacle(x, y) == TrapID:  # Trap
                    view[y, x] = -TrapID
                if self.state.mapInfo.get_obstacle(x, y) == SwampID: # Swamp
                    view[y, x] = -SwampID
                if self.state.mapInfo.gold_amount(x, y) > 0:
                    view[y, x] = self.state.mapInfo.gold_amount(x, y)

        state = view.flatten().tolist()
        # Add position and energy of agent
        state.append(self.state.x)
        state.append(self.state.y)
        state.append(self.state.energy)
        # Add position of bots 
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                state.append(player["posx"])
                state.append(player["posy"])
                
        state = np.array(state)
        return state

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

    def get_9x21x2_state_distinguish(self):
        """
        Same as get_9x21x2_state() but able to distinguish btw swamp and forest

        To be more precise, swamp's -5 and -20 will be deliberately replaced by
        -5.1 and -20.1 to distinguish them from forest's -5 and -20.
        """
        # Note that we must carry the dtype first to float
        state = self.get_9x21x2_state().astype(np.float32)
        ## Here we replace swamp's -5 by -5.1, etc.
        for x in range(constants02.width):
            for y in range(constants02.height):
                if self.state.mapInfo.get_obstacle(x, y) == constants02.terrain_ids["swamp"]:
                    if state[...,0][y, x] == -5:
                        #state[...,0][y, x] = -5.1
                        state[...,0][y, x] = - constants02.replace_swamp5
                    elif state[...,0][y, x] == -20:
                        #state[...,0][y, x] = -20.1
                        state[...,0][y, x] = - constants02.replace_swamp20

        return state


    def get_non_RL_state_01(self):
        """
        return
            view, energy, n_steps, pos_players
            The latter will contain pos of all players, energy and stepCount of agent
        """
        view = self.get_9x21x2_state_distinguish()[...,0]
        pos_players = np.empty((4, 2), dtype=np.int8)
        pos_players[0] = self.state.x, self.state.y
        for i, bot in enumerate(self.socket.bots):
            pos_players[i+1] = bot.info.posx, bot.info.posy
        return view, self.state.energy, self.state.stepCount, pos_players

    def get_non_RL_state_02(self):
        """
        return
            view, energy, n_steps, pos_players
            The latter will contain pos of all players, energy and stepCount of agent
        """
        view = self.get_view_9x21x5()[..., 0]
        pos_players = np.empty((4, 2), dtype=np.int8)
        pos_players[0] = self.state.x, self.state.y
        index = 1
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                x = player["posx"]
                y = player["posy"]
                pos_players[index] = x, y
                index += 1
        return view, self.state.energy, self.state.stepCount, pos_players

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

    def get_reward_6act_bug_01(self):
        # Initialize reward
        reward = 0
        if self.state.status == constants02.agent_state_str2id["out_of_MAP"]:
            #if self.state.stepCount < 50:
            #    reward += -5*(50 - self.state.stepCount)
            reward -= 1000
        #elif self.state.status == constants02.agent_state_str2id["no_more_STEP"]:
        #    #reward += (self.state.score/total_gold) * 100
        #    pass
        elif self.state.status == constants02.agent_state_str2id["no_more_ENERGY"]:
            reward -= 300
        #elif self.state.status == constants02.agent_state_str2id["no_more_GOLD"]:
        #    pass
        #elif self.state.status == constants02.agent_state_str2id["INVALID_action"]:
        #    pass
        else: # Here below: we are almost sure that agent is not out of map
            s = self.get_9x21x2_state()
            try:
                terrain_now = s[self.state.y, self.state.x, 0]
            except Exception as e:
                print(f"{e}")
                print(f"self.state.x, self.state.y = {self.state.x}, {self.state.y} ")
                raise e
            pos_now = np.array([self.state.x, self.state.y])
            reverse_mv = constants02.action_id2ndarray[constants02.reverse_action_id[self.state.lastAction]]
            pos_pre = pos_now + reverse_mv
            x_pre, y_pre = pos_pre
            terrain_pre = s[y_pre, x_pre, 0]

            # Punish `dig on obstacle`
            if self.state.lastAction == constants02.dig:
                if terrain_now < 0:
                    reward -= 100
                elif terrain_now > 0:
                    score_action = self.state.score - self.score_pre
                    reward += score_action
                    self.score_pre = self.state.score
            if self.state.lastAction in (constants02.up, constants02.down, constants02.left, constants02.right,): # i.e. if agent moved
                if terrain_pre > 100: # punish leaving gold
                    reward -= terrain_pre
                if terrain_now > 0: # entering gold
                    if self.state.energy > constants02.punishments["gold"]:
                        reward += 50
                    else:
                        reward -= 100
                if terrain_now < 0: # punish according to terrain_now
                    reward += terrain_now
                    if terrain_now == -100: # i.e. fatal swamp
                        reward -= 500
            if self.state.lastAction == constants02.rest:
                if self.state.energy_pre >= 40:
                    reward -= 200
                if self.state.energy_pre <= 5:
                    reward += 20
            if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                reward += 1

        return reward

    def get_reward_6act_20(self):
        # Initialize reward
        reward = 0
        s = self.get_9x21x2_state()
        try:
            terrain_now = s[self.state.y, self.state.x, 0]
            pos_now = np.array([self.state.x, self.state.y])
            reverse_mv = constants02.action_id2ndarray[constants02.reverse_action_id[self.state.lastAction]]
            pos_pre = pos_now + reverse_mv
            x_pre, y_pre = pos_pre
            terrain_pre = s[y_pre, x_pre, 0]

            # Punish `dig on obstacle`
            if self.state.lastAction == constants02.dig:
                if terrain_now < 0:
                    reward -= 100
                elif terrain_now > 0:
                    score_action = self.state.score - self.score_pre
                    reward += score_action
                    self.score_pre = self.state.score
            if self.state.lastAction in (constants02.up, constants02.down, constants02.left, constants02.right,): # i.e. if agent moved
                if terrain_pre > 100: # punish leaving gold
                    reward -= terrain_pre
                if terrain_now > 0: # entering gold
                    if self.state.energy > constants02.punishments["gold"]:
                        reward += 50
                    else:
                        reward -= 100
                if terrain_now < 0: # punish according to terrain_now
                    reward += terrain_now
                    if terrain_now == -100: # i.e. fatal swamp
                        reward -= 500
            if self.state.lastAction == constants02.rest:
                if self.state.energy_pre >= 40:
                    reward -= 200
                if self.state.energy_pre <= 5:
                    reward += 20
        except Exception as e:
            print(f"{e}")
            print(f"status = {constants02.agent_state_id2str[self.state.status]}")
            print(f"lastAction = {self.state.lastAction}")
            print(f"(x, y) = ({self.state.x}, {self.state.y}) ")
            #raise e
        finally:
            if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                reward += 1
            elif self.state.status == constants02.agent_state_str2id["out_of_MAP"]:
                #if self.state.stepCount < 50:
                #    reward += -5*(50 - self.state.stepCount)
                reward -= 1000
            elif self.state.status == constants02.agent_state_str2id["no_more_ENERGY"]:
                reward -= 300
            #elif self.state.status == constants02.agent_state_str2id["no_more_STEP"]:
            #    #reward += (self.state.score/total_gold) * 100
            #    pass
            #elif self.state.status == constants02.agent_state_str2id["no_more_GOLD"]:
            #    pass
            #elif self.state.status == constants02.agent_state_str2id["INVALID_action"]:
            #    pass
        return reward

    def get_reward_6act_21(self):
        # Initialize reward
        reward = 0
        #s = self.get_9x21x2_state()
        #s = self.get_9x21x2_state_distinguish()
        s = self.get_view_9x21x5()[...,:2]
        pos_now = np.array([self.state.x, self.state.y])
        reverse_mv = constants02.action_id2ndarray[constants02.reverse_action_id[self.state.lastAction]]
        pos_pre = pos_now + reverse_mv
        x_pre, y_pre = pos_pre
        terrain_pre = s[y_pre, x_pre, 0]
        if self.state.status == constants02.agent_state_str2id["out_of_MAP"]:
            #if self.state.stepCount < 50:
            #    reward += -5*(50 - self.state.stepCount)
            reward -= 1000
            if terrain_pre > 0: # punish leaving gold
                reward -= terrain_pre
        else:
            try:
                terrain_now = s[self.state.y, self.state.x, 0]
                #pos_now = np.array([self.state.x, self.state.y])
                #reverse_mv = constants02.action_id2ndarray[constants02.reverse_action_id[self.state.lastAction]]
                #pos_pre = pos_now + reverse_mv
                #x_pre, y_pre = pos_pre
                #terrain_pre = s[y_pre, x_pre, 0]

                # Punish `dig on obstacle`
                if self.state.lastAction == constants02.dig:
                    if terrain_now < 0:
                        reward -= 100
                    elif terrain_now > 0: # TODO: This should be self.terrain_pre, because the amount of gold is altered by lastActions of agent and bots
                        score_action = self.state.score - self.score_pre
                        reward += score_action
                        self.score_pre = self.state.score
                if self.state.lastAction in (constants02.up, constants02.down, constants02.left, constants02.right,): # i.e. if agent moved
                    if terrain_pre > 100: # punish leaving gold
                        reward -= terrain_pre
                    if terrain_now > 0: # entering gold
                        #if self.state.energy_pre > constants02.punishments["gold"]:
                        if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                            ## if upon entering the agent is still alive, we reward it for correctly judging that
                            reward += 50
                        else:
                            reward -= 400
                    if terrain_now < 0: # punish according to terrain_now
                        #logging.debug("(Inside get_reward())")
                        #logging.debug(f"(Inside get_reward()) terrain_now = {terrain_now:.1f} ({self.state.x:2d},{self.state.y:2d})")
                        #logging.debug(f"(Inside get_reward()) terrain_pre = {terrain_pre:.1f} ({x_pre:2d},{y_pre:2d})")
                        reward += terrain_now
                        if terrain_now == -100: # i.e. fatal swamp
                            reward -= 500
                if self.state.lastAction == constants02.rest:
                    if self.state.energy_pre >= 40:
                        reward -= 200
                    if self.state.energy_pre <= 5:
                        reward += 20
                    if terrain_now > 0: # TODO should be self.terrain_pre also
                        if self.state.energy_pre > 15:
                            reward -= 200
            except Exception as e:
                print(f"{e}")
                print(f"step = {self.state.stepCount}")
                print(f"status = {constants02.agent_state_id2str[self.state.status]}")
                print(f"lastAction = {constants02.action_id2str[self.state.lastAction]}")
                print(f"(x, y) = ({self.state.x}, {self.state.y}) ")
                #raise e
            finally:
                if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                    #reward += 1
                    ## Why no reward? Ans: This turned out encouraging agent to "rest"
                    pass
                elif self.state.status == constants02.agent_state_str2id["no_more_ENERGY"]:
                    reward -= 300
                #elif self.state.status == constants02.agent_state_str2id["no_more_STEP"]:
                #    #reward += (self.state.score/total_gold) * 100
                #    pass
                #elif self.state.status == constants02.agent_state_str2id["no_more_GOLD"]:
                #    pass
                #elif self.state.status == constants02.agent_state_str2id["INVALID_action"]:
                #    pass
        return reward

    def get_reward_4act_2channel(self):
        # initialize reward
        reward = 0
        s = self.get_view_9x21x5()[...,:2]
        pos_now = np.array([self.state.x, self.state.y])
        reverse_mv = constants02.action_id2ndarray[constants02.reverse_action_id[self.state.lastAction]]
        pos_pre = pos_now + reverse_mv
        x_pre, y_pre = pos_pre
        terrain_pre = s[y_pre, x_pre, 0]
        if self.state.status == constants02.agent_state_str2id["out_of_MAP"]:
            #if self.state.stepCount < 50:
            #    reward += -5*(50 - self.state.stepCount)
            reward -= 1000
            ## No need to punish leaving gold for 4-act agent
            #if terrain_pre > 0: # punish leaving gold
            #    reward -= terrain_pre
        else:
            try:
                terrain_now = s[self.state.y, self.state.x, 0]

                ## Punish `dig on obstacle`
                #if self.state.lastAction == constants02.dig:
                #    if terrain_now < 0:
                #        reward -= 100
                #    elif terrain_now > 0:
                #        score_action = self.state.score - self.score_pre
                #        reward += score_action
                #        self.score_pre = self.state.score

                if self.state.lastAction in (constants02.up, constants02.down, constants02.left, constants02.right,): # i.e. if agent moved
                    ## No need to punish leaving gold for 4-act agent
                    #if terrain_pre > 100: # punish leaving gold
                    #    reward -= terrain_pre
                    if terrain_now > 0: # entering gold
                        ## It's more intuitive to use the logic of checking whether the agent is still alive
                        #if self.state.energy_pre > constants02.punishments["gold"]:
                        if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                            #reward += 50
                            ## No need to give extra reward to 4act-agent for entering gold 
                            reward += terrain_now
                            self.n_gold_4act_agent += int(terrain_now)
                            self.n_mines_visited += 1
                            # rm all the gold in this grid from the map (to push agent to mv to next gold)
                            for g in self.socket.stepState.golds:
                                    if g.posx == self.state.x and g.posy == self.state.y:
                                        self.socket.stepState.golds.remove(g)
                        else:
                            reward -= 400
                    if terrain_now < 0: # punish according to terrain_now
                        #logging.debug("(Inside get_reward())")
                        #logging.debug(f"(Inside get_reward()) terrain_now = {terrain_now:.1f} ({self.state.x:2d},{self.state.y:2d})")
                        #logging.debug(f"(Inside get_reward()) terrain_pre = {terrain_pre:.1f} ({x_pre:2d},{y_pre:2d})")
                        reward += terrain_now
                        if terrain_now == -100: # i.e. fatal swamp
                            reward -= 500
                #if self.state.lastAction == constants02.rest:
                #    if self.state.energy_pre >= 40:
                #        reward -= 200
                #    if self.state.energy_pre <= 5:
                #        reward += 20
            except Exception as e:
                print(f"{e}")
                print(f"step = {self.state.stepCount}")
                print(f"status = {constants02.agent_state_id2str[self.state.status]}")
                print(f"lastAction = {constants02.action_id2str[self.state.lastAction]}")
                print(f"(x, y) = ({self.state.x}, {self.state.y}) ")
                #raise e
            finally:
                if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                    #reward += 1
                    ## Why no reward? Ans: This turned out encouraging agent to "rest"
                    pass
                elif self.state.status == constants02.agent_state_str2id["no_more_ENERGY"]:
                    reward -= 300
                #elif self.state.status == constants02.agent_state_str2id["no_more_STEP"]:
                #    #reward += (self.state.score/total_gold) * 100
                #    pass
                #elif self.state.status == constants02.agent_state_str2id["no_more_GOLD"]:
                #    pass
                #elif self.state.status == constants02.agent_state_str2id["INVALID_action"]:
                #    pass
        return reward

    def get_reward_5act_2channel(self):
        # initialize reward
        reward = 0
        s = self.get_view_9x21x5()[...,:2]
        pos_now = np.array([self.state.x, self.state.y])
        reverse_mv = constants02.action_id2ndarray[constants02.reverse_action_id[self.state.lastAction]]
        pos_pre = pos_now + reverse_mv
        x_pre, y_pre = pos_pre
        terrain_pre = s[y_pre, x_pre, 0]
        if self.state.status == constants02.agent_state_str2id["out_of_MAP"]:
            #if self.state.stepCount < 50:
            #    reward += -5*(50 - self.state.stepCount)
            reward -= 1000
            ## No need to punish leaving gold for 5-act agent, either
            #if terrain_pre > 0: # punish leaving gold
            #    reward -= terrain_pre
        else:
            try:
                terrain_now = s[self.state.y, self.state.x, 0]

                ## Punish `dig on obstacle`
                #if self.state.lastAction == constants02.dig:
                #    if terrain_now < 0:
                #        reward -= 100
                #    elif terrain_now > 0:
                #        score_action = self.state.score - self.score_pre
                #        reward += score_action
                #        self.score_pre = self.state.score

                if self.state.lastAction in (constants02.up, constants02.down, constants02.left, constants02.right,): # i.e. if agent moved
                    ## No need to punish leaving gold for 5-act agent, either
                    #if terrain_pre > 100: # punish leaving gold
                    #    reward -= terrain_pre
                    if terrain_now > 0: # i.e. entering gold
                        ## It's more intuitive to use the logic of checking whether the agent is still alive
                        #if self.state.energy_pre > constants02.punishments["gold"]:
                        if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                            #reward += 50
                            ## No need to give extra reward to 4act-agent for entering gold 
                            reward += terrain_now
                            self.n_gold_5act_agent += int(terrain_now)
                            self.n_mines_visited += 1
                            # rm all the gold in this grid from the map (to push agent to mv to next gold)
                            for g in self.socket.stepState.golds:
                                    if g.posx == self.state.x and g.posy == self.state.y:
                                        self.socket.stepState.golds.remove(g)
                        else:
                            reward -= 400
                    if terrain_now < 0: # punish according to terrain_now
                        #logging.debug("(Inside get_reward())")
                        #logging.debug(f"(Inside get_reward()) terrain_now = {terrain_now:.1f} ({self.state.x:2d},{self.state.y:2d})")
                        #logging.debug(f"(Inside get_reward()) terrain_pre = {terrain_pre:.1f} ({x_pre:2d},{y_pre:2d})")
                        reward += terrain_now
                        if terrain_now == -100: # i.e. fatal swamp
                            reward -= 500
                if self.state.lastAction == constants02.rest:
                    if self.state.energy_pre > 40:
                        reward -= 200
                    if self.state.energy_pre <= 5:
                        reward += 100
                    if terrain_now > 0: # TODO should be self.terrain_pre also
                        if self.state.energy_pre > 15:
                            reward -= 200
            except Exception as e:
                print(f"{e}")
                print(f"step = {self.state.stepCount}")
                print(f"status = {constants02.agent_state_id2str[self.state.status]}")
                print(f"lastAction = {constants02.action_id2str[self.state.lastAction]}")
                print(f"(x, y) = ({self.state.x}, {self.state.y}) ")
                #raise e
            finally:
                if self.state.status == constants02.agent_state_str2id["PLAYing"]:
                    #reward += 1
                    ## Why no reward? Ans: This turned out encouraging agent to "rest"
                    pass
                elif self.state.status == constants02.agent_state_str2id["no_more_ENERGY"]:
                    reward -= 300
                #elif self.state.status == constants02.agent_state_str2id["no_more_STEP"]:
                #    #reward += (self.state.score/total_gold) * 100
                #    pass
                #elif self.state.status == constants02.agent_state_str2id["no_more_GOLD"]:
                #    pass
                #elif self.state.status == constants02.agent_state_str2id["INVALID_action"]:
                #    pass
        return reward


    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        if self.state.status != State.STATUS_PLAYING:
            logging.debug(f"check_terminate: {agent_state_id2str[self.state.status]}")
        return self.state.status != State.STATUS_PLAYING


