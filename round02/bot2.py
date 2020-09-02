import numpy as np

from miner_state import State
import non_RL_agent
import non_RL_agent02
import non_RL_agent03
import non_RL_agent06
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


class Bot2:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5
    non_RL_policies = [non_RL_agent06.greedy_policy]

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        self.policy = np.random.choice(self.non_RL_policies)

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

    def next_action(self):
        s = self.get_198_state()
        return int(self.policy(s))

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

    #def next_action(self):
    #    if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
    #        if self.info.energy >= 6:
    #            return self.ACTION_CRAFT
    #        else:
    #            return self.ACTION_FREE
    #    if self.info.energy < 5:
    #        return self.ACTION_FREE
    #    else:
    #        action = np.random.randint(0, 4)
    #        return action
