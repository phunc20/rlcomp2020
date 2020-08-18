import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State


TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score#Storing the last score for designing the reward function
        self.pos_x_pre =  self.state.x
        self.pos_y_pre = self.state.y
        
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
            
            self.score_pre = self.state.score#Storing the last score for designing the reward function
            self.pos_x_pre =  self.state.x
            self.pos_y_pre = self.state.y
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

    # Functions are customized by client
    def get_state(self):
        #Local view
        view =  np.zeros([5,5])
        for i in range(-2,3):
            for j in range(-2,3):
                index_x = self.state.x + i
                index_y = self.state.y + j
                if index_x < 0 or index_y < 0 or index_x >= self.state.mapInfo.max_x or index_y >= self.state.mapInfo.max_y:
                    view[2+i,2+j] = -1
                else:
                        if self.state.mapInfo.get_obstacle(index_x, index_y) == TreeID:
                            view[2+i,2+j] = -1
                        if self.state.mapInfo.get_obstacle(index_x, index_y) == TrapID:
                            #view[2+i,2+j] = -1
                            pass
                        if self.state.mapInfo.get_obstacle(index_x, index_y) == SwampID:
                            view[2+i,2+j] = -1
                                
        #Create the state
        DQNState = view.flatten().tolist()
        self.pos_x_gold_first =self.state.x
        self.pos_y_gold_first = self.state.y
        if len(self.state.mapInfo.golds) > 0:
            self.pos_x_gold_first = self.state.mapInfo.golds[0]["posx"]
            self.pos_y_gold_first = self.state.mapInfo.golds[0]["posy"]         
        DQNState.append(self.pos_x_gold_first - self.state.x)
        DQNState.append(self.pos_y_gold_first - self.state.y)
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        goldamount = self.state.mapInfo.gold_amount(self.state.x, self.state.y)
        if goldamount > 0:
            # i.e. when we (agent) are standing on gold
            reward += 10
            #remove the gold
            for g in self.socket.stepState.golds:
                    if g.posx == self.state.x and g.posy == self.state.y:
                        self.socket.stepState.golds.remove(g)
            
        #If the DQN agent crashs into obstacles (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= 0.2
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            #reward -= 0.2
            pass
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= 0.4
        # previous distance and current distance, i.e. distance to 1st gold, previously and currently
        #dis_pre = np.sqrt((self.pos_x_pre- self.pos_x_gold_first)**2 + (self.pos_y_pre-self.pos_y_gold_first)**2)
        dis_pre = np.linalg.norm([self.pos_x_pre - self.pos_x_gold_first, self.pos_y_pre - self.pos_y_gold_first])
        #dis_curr = np.sqrt((self.state.x - self.pos_x_gold_first)**2 + (self.state.y - self.pos_y_gold_first)**2)
        dis_curr = np.linalg.norm([self.state.x - self.pos_x_gold_first, self.state.y - self.pos_y_gold_first])
        #if (dis_curr - dis_pre) <= 0: # Reducing the distance , reward ++
        if dis_curr < dis_pre: # Reducing the distance , reward ++
                reward += 0.1
        else:
                reward -= 0.1
        
        # If out of the map, then the DQN agent should be punished by a large nagative reward
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward -= 10
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
