from miner_state import State
import numpy as np
from revived_non_RL_01 import Goat
import constants02


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

class Bot3:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        self.goat = Goat()
        self.policy = self.goat.policy_nearest_gold


    def get_198_state(self):
        view = np.zeros([self.state.mapInfo.max_y + 1, self.state.mapInfo.max_x + 1], dtype=int)
        for x in range(self.state.mapInfo.max_x + 1):
            for y in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(x, y) == constants02.terrain_ids["forest"]:  # Tree
                    view[y, x] = - constants02.terrain_ids["forest"]
                if self.state.mapInfo.get_obstacle(x, y) == constants02.terrain_ids["trap"]:
                    view[y, x] = - constants02.terrain_ids["trap"]
                if self.state.mapInfo.get_obstacle(x, y) == constants02.terrain_ids["swamp"]:
                    view[y, x] = - constants02.terrain_ids["swamp"]
                if self.state.mapInfo.gold_amount(x, y) > 0:
                    view[y, x] = self.state.mapInfo.gold_amount(x, y)

        state = view.flatten().tolist()
        
        #state.append(self.state.x)
        #state.append(self.state.y)
        #state.append(self.state.energy)
        state.append(self.info.posx)
        state.append(self.info.posy)
        state.append(self.info.energy)
        for player in self.state.players:
            # self.info.playerId is the id of the current bot
            if player["playerId"] != self.info.playerId:
                state.append(player["posx"])
                state.append(player["posy"])
                
        state = np.array(state)
        return state

    #def next_action(self):
    #    #s = self.get_198_state()
    #    view, energy, stepCount, pos_players = env.get_non_RL_state_02()
    #    #return int(self.policy(s))
    #    int_action = int(self.policy(view, energy, pos_players))
    #    #print(f"(bot1) pos ({self.info.posx:2d},{self.info.posy:2d}) energy {self.info.energy:2d}")
    #    #print(f"       {constants02.action_id2str[int_action]}")
    #    return int_action

    def next_action(self):
        if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
            if self.info.energy >= 6:
                return self.ACTION_CRAFT
            else:
                return self.ACTION_FREE
        if self.info.energy < 5:
            return self.ACTION_FREE
        else:
            action = self.ACTION_GO_LEFT
            if self.info.posx % 2 == 0:
                if self.info.posy < self.state.mapInfo.max_y:
                    action = self.ACTION_GO_DOWN
            else:
                if self.info.posy > 0:
                    action = self.ACTION_GO_UP
                else:
                    action = self.ACTION_GO_RIGHT
            return action

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

    #def get_score(self):
    #    return [player["score"] for player in minerEnv.socket.bots[1].state.players if player["playerId"] == self.info.playerId][0]
