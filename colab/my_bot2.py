from MINER_STATE import State
import numpy as np
import constants


class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0

#########################
##  Blah Blah Blah  ##
#########################

class Bot2:

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        
    def get_state(self):
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

        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        #DQNState.append(self.state.x)
        #DQNState.append(self.state.y)
        #DQNState.append(self.state.energy)
        DQNState.append(self.info.posx)
        DQNState.append(self.info.posy)
        DQNState.append(self.info.energy)
        for player in self.state.players:
            # self.info.playerId is the id of the current bot
            if player["playerId"] != self.info.playerId:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])
                
        DQNState = np.array(DQNState)

        return DQNState


    def next_action(self):
        s = self.get_state()
        #return greedy_policy(s, how_gold=find_largest_gold) 
        return int(constants.greedy_policy(s, how_gold=constants.find_worthiest_gold))

    def get_score(self):
        return [player["score"] for player in minerEnv.socket.bots[1].state.players if player["playerId"] == self.info.playerId][0]


    def new_game(self, data):
        try:
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
