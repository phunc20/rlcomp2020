from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
from GAME_SOCKET import GameSocket #in testing version, please use GameSocket instead of GameSocketDummy
from MINER_STATE import State

TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score#Storing the last score for designing the reward function
        
    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            print(message)
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            #print("New state: ", message)
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
                            view[2+i,2+j] = -1
                        if self.state.mapInfo.get_obstacle(index_x, index_y) == SwampID:
                            view[2+i,2+j] = -1
                                
        #Create the state
        DQNState = view.flatten().tolist()
        self.pos_x_gold_first =self.state.x
        self.pos_y_gold_first = self.state.y
        if len(self.state.mapInfo.golds) > 0:
            self.pos_x_gold_first = self.state.mapInfo.golds[0]["posx"]
            self.pos_y_gold_first = self.state.mapInfo.golds[0]["posy"]         
        DQNState.append(self.pos_x_gold_first-self.state.x)
        DQNState.append(self.pos_y_gold_first-self.state.y)
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)
        
        return DQNState

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
