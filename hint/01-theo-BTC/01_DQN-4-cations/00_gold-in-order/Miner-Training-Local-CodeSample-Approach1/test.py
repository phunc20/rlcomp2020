import matplotlib.pyplot as plt
import numpy as np
#from viz_utils import *

import sys
import numpy as np
#import pandas as pd
import datetime
import json
from array import *
import os
import math
from random import randrange
import random

#from keras.models import Sequential
#from keras.models import model_from_json
#from keras.layers import Dense, Activation
#from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

import tensorflow.keras as keras

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
tf.disable_v2_behavior()

import constants
import non_RL_agent
import non_RL_agent3
from DQNModel import DQN
#from MinerEnv import MinerEnv

#Classes in GAME_SOCKET_DUMMY.py
class ObstacleInfo:
    # initial energy for obstacles: Land (key = 0): -1, Forest(key = -1): 0 (random), Trap(key = -2): -10, Swamp (key = -3): -5
    types = {0: -1, -1: 0, -2: -10, -3: -5}

    def __init__(self):
        self.type = 0
        self.posx = 0
        self.posy = 0
        self.value = 0
        
class GoldInfo:
    def __init__(self):
        self.posx = 0
        self.posy = 0
        self.amount = 0

    def loads(self, data):
        golds = []
        for gd in data:
            g = GoldInfo()
            g.posx = gd["posx"]
            g.posy = gd["posy"]
            g.amount = gd["amount"]
            golds.append(g)
        return golds

class PlayerInfo:
    STATUS_PLAYING = 0
    STATUS_ELIMINATED_WENT_OUT_MAP = 1
    STATUS_ELIMINATED_OUT_OF_ENERGY = 2
    STATUS_ELIMINATED_INVALID_ACTION = 3
    STATUS_STOP_EMPTY_GOLD = 4
    STATUS_STOP_END_STEP = 5

    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = PlayerInfo.STATUS_PLAYING
        self.freeCount = 0

class GameInfo:
    def __init__(self):
        self.numberOfPlayers = 1
        self.width = 0
        self.height = 0
        self.steps = 100
        self.golds = []
        self.obstacles = []

    def loads(self, data):
        m = GameInfo()
        m.width = data["width"]
        m.height = data["height"]
        m.golds = GoldInfo().loads(data["golds"])
        m.obstacles = data["obstacles"]
        m.numberOfPlayers = data["numberOfPlayers"]
        m.steps = data["steps"]
        return m

class UserMatch:
    def __init__(self):
        self.playerId = 1
        self.posx = 0
        self.posy = 0
        self.energy = 50
        self.gameinfo = GameInfo()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class StepState:
    def __init__(self):
        self.players = []
        self.golds = []
        self.changedObstacles = []

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)




#Main class in GAME_SOCKET_DUMMY.py
class GameSocket:
    bog_energy_chain = {-5: -20, -20: -40, -40: -100, -100: -100}

    def __init__(self):
        self.stepCount = 0
        self.maxStep = 0
        self.mapdir = "Maps"  # where to load all pre-defined maps
        self.mapid = ""
        self.userMatch = UserMatch()
        self.user = PlayerInfo(1)
        self.stepState = StepState()
        self.maps = {}  # key: map file name, value: file content
        self.map = []  # running map info: 0->Land, -1->Forest, -2->Trap, -3:Swamp, >0:Gold
        self.energyOnMap = []  # self.energyOnMap[x][y]: <0, amount of energy which player will consume if it move into (x,y)
        self.E = 50
        self.resetFlag = True
        self.craftUsers = []  # players that craft at current step - for calculating amount of gold
        self.bots = []
        self.craftMap = {}  # cells that players craft at current step, key: x_y, value: number of players that craft at (x,y)

    def init_bots(self):
        self.bots = [Bot1(2), Bot2(3), Bot3(4)]  # use bot1(id=2), bot2(id=3), bot3(id=4)
        #for (bot) in self.bots:  # at the beginning, all bots will have same position, energy as player
        for bot in self.bots:  # at the beginning, all bots will have same position, energy as player
            bot.info.posx = self.user.posx
            bot.info.posy = self.user.posy
            bot.info.energy = self.user.energy
            bot.info.lastAction = -1
            bot.info.status = PlayerInfo.STATUS_PLAYING
            bot.info.score = 0
            self.stepState.players.append(bot.info)
        self.userMatch.gameinfo.numberOfPlayers = len(self.stepState.players)
        #print("numberOfPlayers: ", self.userMatch.gameinfo.numberOfPlayers)

    def reset(self, requests):  # load new game by given request: [map id (filename), posx, posy, initial energy]
        # load new map
        self.reset_map(requests[0])
        self.userMatch.posx = int(requests[1])
        self.userMatch.posy = int(requests[2])
        self.userMatch.energy = int(requests[3])
        self.userMatch.gameinfo.steps = int(requests[4])
        self.maxStep = self.userMatch.gameinfo.steps

        # init data for players
        self.user.posx = self.userMatch.posx  # in
        self.user.posy = self.userMatch.posy
        self.user.energy = self.userMatch.energy
        self.user.status = PlayerInfo.STATUS_PLAYING
        self.user.score = 0
        self.stepState.players = [self.user]
        self.E = self.userMatch.energy
        self.resetFlag = True
        self.init_bots()
        self.stepCount = 0

    def reset_map(self, id):  # load map info
        self.mapId = id
        self.map = json.loads(self.maps[self.mapId])
        self.userMatch = self.map_info(self.map)
        self.stepState.golds = self.userMatch.gameinfo.golds
        self.map = json.loads(self.maps[self.mapId])
        self.energyOnMap = json.loads(self.maps[self.mapId])
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                if self.map[x][y] > 0:  # gold
                    self.energyOnMap[x][y] = -4
                else:  # obstacles
                    self.energyOnMap[x][y] = ObstacleInfo.types[self.map[x][y]]

    def connect(self): # simulate player's connect request
        print("Connected to server.")
        for mapid in range(len(Maps)):
            filename = "map" + str(mapid)
            print("Found: " + filename)
            self.maps[filename] = str(Maps[mapid])

    def map_info(self, map):  # get map info
        # print(map)
        userMatch = UserMatch()
        userMatch.gameinfo.height = len(map)
        userMatch.gameinfo.width = len(map[0])
        i = 0
        while i < len(map):
            j = 0
            while j < len(map[i]):
                if map[i][j] > 0:  # gold
                    g = GoldInfo()
                    g.posx = j
                    g.posy = i
                    g.amount = map[i][j]
                    userMatch.gameinfo.golds.append(g)
                else:  # obstacles
                    o = ObstacleInfo()
                    o.posx = j
                    o.posy = i
                    o.type = -map[i][j]
                    o.value = ObstacleInfo.types[map[i][j]]
                    userMatch.gameinfo.obstacles.append(o)
                j += 1
            i += 1
        return userMatch

    def receive(self):  # send data to player (simulate player's receive request)
        if self.resetFlag:  # for the first time -> send game info
            self.resetFlag = False
            data = self.userMatch.to_json()
            for (bot) in self.bots:
                bot.new_game(data)
            # print(data)
            return data
        else:  # send step state
            self.stepCount = self.stepCount + 1
            if self.stepCount >= self.maxStep:
                for player in self.stepState.players:
                    player.status = PlayerInfo.STATUS_STOP_END_STEP
            data = self.stepState.to_json()
            #for (bot) in self.bots:  # update bots' state
            for bot in self.bots:  # update bots' state
                bot.new_state(data)
            # print(data)
            return data

    def send(self, message):  # receive message from player (simulate send request from player)
        if message.isnumeric():  # player send action
            self.resetFlag = False
            self.stepState.changedObstacles = []
            action = int(message)
            # print("Action = ", action)
            self.user.lastAction = action
            self.craftUsers = []
            self.step_action(self.user, action)
            for bot in self.bots:
                if bot.info.status == PlayerInfo.STATUS_PLAYING:
                    action = bot.next_action()
                    bot.info.lastAction = action
                    # print("Bot Action: ", action)
                    self.step_action(bot.info, action)
            self.action_5_craft()
            for c in self.stepState.changedObstacles:
                self.map[c["posy"]][c["posx"]] = -c["type"]
                self.energyOnMap[c["posy"]][c["posx"]] = c["value"]

        else:  # reset game
            requests = message.split(",")
            #print("Reset game: ", requests[:3], end='')
            self.reset(requests)

    def step_action(self, user, action):
        switcher = {
            0: self.action_0_left,
            1: self.action_1_right,
            2: self.action_2_up,
            3: self.action_3_down,
            4: self.action_4_free,
            5: self.action_5_craft_pre
        }
        func = switcher.get(action, self.invalidAction)
        func(user)

    def action_5_craft_pre(self, user):  # collect players who craft at current step
        user.freeCount = 0
        if self.map[user.posy][user.posx] <= 0:  # craft at the non-gold cell
            user.energy -= 10
            if user.energy <= 0:
                user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
                user.lastAction = 6 #eliminated
        else:
            user.energy -= 5
            if user.energy > 0:
                self.craftUsers.append(user)
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    count = self.craftMap[key]
                    self.craftMap[key] = count + 1
                else:
                    self.craftMap[key] = 1
            else:
                user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
                user.lastAction = 6 #eliminated

    def action_0_left(self, user):  # user go left
        user.freeCount = 0
        user.posx = user.posx - 1
        if user.posx < 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_1_right(self, user):  # user go right
        user.freeCount = 0
        user.posx = user.posx + 1
        if user.posx >= self.userMatch.gameinfo.width:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_2_up(self, user):  # user go up
        user.freeCount = 0
        user.posy = user.posy - 1
        if user.posy < 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_3_down(self, user):  # user go right
        user.freeCount = 0
        user.posy = user.posy + 1
        if user.posy >= self.userMatch.gameinfo.height:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_4_free(self, user):  # user free
        user.freeCount += 1
        if user.freeCount == 1:
            user.energy += int(self.E / 4)
        elif user.freeCount == 2:
            user.energy += int(self.E / 3)
        elif user.freeCount == 3:
            user.energy += int(self.E / 2)
        else:
            user.energy = self.E
        if user.energy > self.E:
            user.energy = self.E

    def action_5_craft(self):
        craftCount = len(self.craftUsers)
        # print ("craftCount",craftCount)
        if (craftCount > 0):
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                c = self.craftMap[key]
                m = min(math.ceil(self.map[y][x] / c), 50)
                user.score += m
                # print ("user", user.playerId, m)
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    c = self.craftMap[key]
                    del self.craftMap[key]
                    m = min(math.ceil(self.map[y][x] / c), 50)
                    self.map[y][x] -= m * c
                    if self.map[y][x] < 0:
                        self.map[y][x] = 0
                        self.energyOnMap[y][x] = ObstacleInfo.types[0]
                    for g in self.stepState.golds:
                        if g.posx == x and g.posy == y:
                            g.amount = self.map[y][x]
                            if g.amount == 0:
                                self.stepState.golds.remove(g)
                                self.add_changed_obstacle(x, y, 0, ObstacleInfo.types[0])
                                if len(self.stepState.golds) == 0:
                                    for player in self.stepState.players:
                                        player.status = PlayerInfo.STATUS_STOP_EMPTY_GOLD
                            break;
            self.craftMap = {}

    def invalidAction(self, user):
        user.status = PlayerInfo.STATUS_ELIMINATED_INVALID_ACTION
        user.lastAction = 6 #eliminated

    def go_to_pos(self, user):  # player move to cell(x,y)
        if self.map[user.posy][user.posx] == -1:
            user.energy -= randrange(16) + 5
        elif self.map[user.posy][user.posx] == 0:
            user.energy += self.energyOnMap[user.posy][user.posx]
        elif self.map[user.posy][user.posx] == -2:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 0, ObstacleInfo.types[0])
        elif self.map[user.posy][user.posx] == -3:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 3,
                                      self.bog_energy_chain[self.energyOnMap[user.posy][user.posx]])
        else:
            user.energy -= 4
        if user.energy <= 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
            user.lastAction = 6 #eliminated

    def add_changed_obstacle(self, x, y, t, v):
        added = False
        for o in self.stepState.changedObstacles:
            if o["posx"] == x and o["posy"] == y:
                added = True
                break
        if added == False:
            o = {}
            o["posx"] = x
            o["posy"] = y
            o["type"] = t
            o["value"] = v
            self.stepState.changedObstacles.append(o)

    def close(self):
        print("Close socket.")



class Bot1:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

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
        #return int(greedy_policy(s))
        return int(non_RL_agent.greedy_policy(s))

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


class Bot2:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

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
        return int(non_RL_agent3.greedy_policy(s))
 
        #if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
        #    if self.info.energy >= 6:
        #        return self.ACTION_CRAFT
        #    else:
        #        return self.ACTION_FREE
        #if self.info.energy < 5:
        #    return self.ACTION_FREE
        #else:
        #    action = np.random.randint(0, 4)            
        #    return action


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
    
    def get_score(self):
        return [player["score"] for player in minerEnv.socket.bots[1].state.players if player["playerId"] == self.info.playerId][0]


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
            
    def get_score(self):
        return [player["score"] for player in minerEnv.socket.bots[1].state.players if player["playerId"] == self.info.playerId][0]



#MinerState.py
def str_2_json(str):
    return json.loads(str, encoding="utf-8")


class MapInfo:
    def __init__(self):
        self.max_x = 0 #Width of the map
        self.max_y = 0 #Height of the map
        self.golds = [] #List of the golds in the map
        self.obstacles = []
        self.numberOfPlayers = 0
        self.maxStep = 0 #The maximum number of step is set for this map

    def init_map(self, gameInfo):
        #Initialize the map at the begining of each episode
        self.max_x = gameInfo["width"] - 1
        self.max_y = gameInfo["height"] - 1
        self.golds = gameInfo["golds"]
        self.obstacles = gameInfo["obstacles"]
        self.maxStep = gameInfo["steps"]
        self.numberOfPlayers = gameInfo["numberOfPlayers"]

    def update(self, golds, changedObstacles):
        #Update the map after every step
        self.golds = golds
        for cob in changedObstacles:
            newOb = True
            for ob in self.obstacles:
                if cob["posx"] == ob["posx"] and cob["posy"] == ob["posy"]:
                    newOb = False
                    #print("cell(", cob["posx"], ",", cob["posy"], ") change type from: ", ob["type"], " -> ",
                    #      cob["type"], " / value: ", ob["value"], " -> ", cob["value"])
                    ob["type"] = cob["type"]
                    ob["value"] = cob["value"]
                    break
            if newOb:
                self.obstacles.append(cob)
                #print("new obstacle: ", cob["posx"], ",", cob["posy"], ", type = ", cob["type"], ", value = ",
                #      cob["value"])

    def get_min_x(self):
        return min([cell["posx"] for cell in self.golds])

    def get_max_x(self):
        return max([cell["posx"] for cell in self.golds])

    def get_min_y(self):
        return min([cell["posy"] for cell in self.golds])

    def get_max_y(self):
        return max([cell["posy"] for cell in self.golds])

    def is_row_has_gold(self, y):
        return y in [cell["posy"] for cell in self.golds]

    def is_column_has_gold(self, x):
        return x in [cell["posx"] for cell in self.golds]

    def gold_amount(self, x, y): #Get the amount of golds at cell (x,y)
        for cell in self.golds:
            if x == cell["posx"] and y == cell["posy"]:
                return cell["amount"]
        return 0 

    def get_obstacle(self, x, y):  # Get the kind of the obstacle at cell(x,y)
        for cell in self.obstacles:
            if x == cell["posx"] and y == cell["posy"]:
                return cell["type"]
        return -1  # No obstacle at the cell (x,y)


class State:
    STATUS_PLAYING = 0
    STATUS_ELIMINATED_WENT_OUT_MAP = 1
    STATUS_ELIMINATED_OUT_OF_ENERGY = 2
    STATUS_ELIMINATED_INVALID_ACTION = 3
    STATUS_STOP_EMPTY_GOLD = 4
    STATUS_STOP_END_STEP = 5

    def __init__(self):
        self.end = False
        self.score = 0
        self.lastAction = None
        self.id = 0
        self.x = 0
        self.y = 0
        self.energy = 0
        self.mapInfo = MapInfo()
        self.players = []
        self.stepCount = 0
        self.status = State.STATUS_PLAYING

    def init_state(self, data): #parse data from server into object
        game_info = str_2_json(data)
        self.end = False
        self.score = 0
        self.lastAction = None
        self.id = game_info["playerId"]
        self.x = game_info["posx"]
        self.y = game_info["posy"]
        self.energy = game_info["energy"]
        self.mapInfo.init_map(game_info["gameinfo"])
        self.stepCount = 0
        self.status = State.STATUS_PLAYING
        self.players = [{"playerId": 2, "posx": self.x, "posy": self.y},
                        {"playerId": 3, "posx": self.x, "posy": self.y},
                        {"playerId": 4, "posx": self.x, "posy": self.y}]

    def update_state(self, data):
        new_state = str_2_json(data)
        for player in new_state["players"]:
            if player["playerId"] == self.id:
                self.x = player["posx"]
                self.y = player["posy"]
                self.energy = player["energy"]
                self.score = player["score"]
                self.lastAction = player["lastAction"]
                self.status = player["status"]

        self.mapInfo.update(new_state["golds"], new_state["changedObstacles"])
        self.players = new_state["players"]
        for i in range(len(self.players), 4, 1):
            self.players.append({"playerId": i, "posx": self.x, "posy": self.y})
        self.stepCount = self.stepCount + 1


# In[498]:


##MinerEnv.py
TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self, host, port):
        #self.socket = GameSocket(host, port)
        self.socket = GameSocket()
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
        self.pos_x_gold_first = self.state.x
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
            # i.e. when we (agent) are standing on gold, or when we finally arrive at gold
            reward += 10
            # remove the gold
            for g in self.socket.stepState.golds:
                    if g.posx == self.state.x and g.posy == self.state.y:
                        self.socket.stepState.golds.remove(g)
        #gold_digged = self.state.score - self.score_pre
        #self.score_pre = self.state.score
        #if gold_digged > 0:
        #    #reward += 10
        #    reward += 0.1*(constants.n_allowed_steps - self.state.stepCount) + 5
            
        #If the DQN agent crashs into obstacles (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= 1
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= 0.5
            #pass
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= 40
        # previous distance and current distance, i.e. distance to 1st gold, previously and currently
        #dis_pre = np.sqrt((self.pos_x_pre- self.pos_x_gold_first)**2 + (self.pos_y_pre-self.pos_y_gold_first)**2)
        dis_pre = np.linalg.norm([self.pos_x_pre - self.pos_x_gold_first, self.pos_y_pre - self.pos_y_gold_first])
        #dis_curr = np.sqrt((self.state.x - self.pos_x_gold_first)**2 + (self.state.y - self.pos_y_gold_first)**2)
        dis_curr = np.linalg.norm([self.state.x - self.pos_x_gold_first, self.state.y - self.pos_y_gold_first])
        #if (dis_curr - dis_pre) <= 0: # Reducing the distance , reward ++
        if dis_curr < dis_pre: # Reducing the distance , reward ++
                reward += 1
        else:
                reward -= 1
        
        # If out of the map, then the DQN agent should be punished by a large nagative reward
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward -= 40
        if self.state.status == State.STATUS_PLAYING:
            reward += 0.5

        # Meaningless rest: punish
        if self.state.lastAction == constants.rest and self.state.energy > 42:
            reward -= 3
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
#TreeID = 1
#TrapID = 2
#SwampID = 3
#class MinerEnv:
#    def __init__(self):
#        self.socket = GameSocket()
#        self.state = State()
#        
#        self.score_pre = self.state.score#Storing the last score for designing the reward function
#
#    def start(self): #connect to server
#        self.socket.connect()
#
#    def end(self): #disconnect server
#        self.socket.close()
#
#    def send_map_info(self, request):#tell server which map to run
#        self.socket.send(request)
#
#    def reset(self): #start new game
#        try:
#            message = self.socket.receive() #receive game info from server
#            self.state.init_state(message) #init state
#        except Exception as e:
#            import traceback
#            traceback.print_exc()
#
#    def step(self, action): #step process
#        self.socket.send(action) #send action to server
#        try:
#            message = self.socket.receive() #receive new state from server
#            self.state.update_state(message) #update to local state
#        except Exception as e:
#            import traceback
#            traceback.print_exc()
#
#    # Functions are customized by client
#    def get_state(self):
#        # Building the map
#        #view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
#        view = np.zeros([self.state.mapInfo.max_y + 1, self.state.mapInfo.max_x + 1], dtype=int)
#        for x in range(self.state.mapInfo.max_x + 1):
#            for y in range(self.state.mapInfo.max_y + 1):
#                if self.state.mapInfo.get_obstacle(x, y) == TreeID:  # Tree
#                    view[y, x] = -TreeID
#                if self.state.mapInfo.get_obstacle(x, y) == TrapID:  # Trap
#                    view[y, x] = -TrapID
#                if self.state.mapInfo.get_obstacle(x, y) == SwampID: # Swamp
#                    view[y, x] = -SwampID
#                if self.state.mapInfo.gold_amount(x, y) > 0:
#                    view[y, x] = self.state.mapInfo.gold_amount(x, y)
#
#        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
#        
#        # Add position and energy of agent to the DQNState
#        DQNState.append(self.state.x)
#        DQNState.append(self.state.y)
#        DQNState.append(self.state.energy)
#        #Add position of bots 
#        for player in self.state.players:
#            if player["playerId"] != self.state.id:
#                DQNState.append(player["posx"])
#                DQNState.append(player["posy"])
#                
#        #Convert the DQNState from list to array for training
#        DQNState = np.array(DQNState)
#
#        return DQNState
#
#    def get_reward(self):
#        # Calculate reward
#        reward = 0
#        score_action = self.state.score - self.score_pre
#        self.score_pre = self.state.score
#        if score_action > 0:
#            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
#            #reward += score_action
#            reward += score_action*5
#            
#        ##If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
#        #if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
#        #    reward -= TreeID
#        #if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
#        #    reward -= TrapID
#        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
#            reward -= SwampID
#            if self.state.lastAction == 4:
#                reward -= 40
#
#        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
#        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
#            #if self.state.stepCount < 50:
#            #    reward += -5*(50 - self.state.stepCount)
#            reward += -50
#            
#        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
#        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
#            if self.state.stepCount < 50:
#                reward += -(50 - self.state.stepCount)
#                if self.state.lastAction != 4:
#                    # 4 is taking a rest
#                    reward += -10
#        
#        # control comes to here \implies our agent is not dead yet
#        if self.state.status == State.STATUS_PLAYING:
#            if self.state.energy >= 45 and self.state.lastAction == 4:
#                reward -= 30
#        # print ("reward",reward)
#        return reward
#
#    def check_terminate(self):
#        #Checking the status of the game
#        #it indicates the game ends or is playing
#        return self.state.status != State.STATUS_PLAYING


# Maps of 1st round
def CreateMaps():
    map0 = [
      [450,-2,0,-2,150,-1,0,0,0,0,-1,-2,-2,-2,0,0,0,0,150,-2,350],
      [-2,-2,-2,-2,-1,0,-1,-1,-1,-1,-3,50,-2,-2,-2,-2,-3,-3,50,-2,-1],
      [-2,-2,200,-2,0,-2,0,-2,-3,-3,-2,0,-3,-2,-2,150,-3,-3,0,0,50],
      [0,-3,-3,-2,0,0,-1,0,550,-3,-2,0,0,0,-1,0,0,-1,-1,-1,-2],
      [-2,0,0,0,-1,0,-1,50,300,-3,-2,0,-3,0,0,0,-1,-3,-3,-2,-1],
      [-1,-3,-1,-3,0,-2,0,0,-2,-1,100,-3,0,-2,300,-3,0,-2,-3,-2,0],
      [-2,-3,-1,-3,-1,500,-1,-3,-2,-1,0,-1,0,-1,0,-1,0,-2,-3,-3,-1],
      [0,-3,-1,-3,0,-2,-3,-3,0,0,0,0,-2,0,-2,-3,-3,-3,-3,200,-1],
      [1200,-3,-1,-3,-1,-1,-2,-2,0,-1,150,-2,0,-2,0,0,-2,-3,-3,1500,50],
    ]
    map1 = [
      [550,-1,-1,-2,-2,-2,-2,150,0,0,0,-3,-2,400,-3,650,-2,-1,0,0,0],
      [350,-1,-1,-2,0,0,-2,-2,-2,-1,0,-3,-3,-2,300,-2,-1,650,-2,-1,0],
      [-1,-1,450,-2,0,0,0,0,0,0,0,0,0,-3,-3,-3,-1,-1,-3,0,0],
      [-1,-1,-2,-2,0,-3,-2,0,0,-3,-3,-3,0,0,0,0,0,0,-3,0,150],
      [0,0,-2,0,-3,-3,-3,0,-1,-2,400,-3,-3,0,-2,0,-1,0,-3,0,0],
      [0,200,-2,0,-3,250,-1,0,0,-1,-2,-3,-2,0,-1,300,-1,0,-3,-1,0],
      [-3,-3,-2,0,-3,-3,-3,0,0,0,0,-3,-3,-3,-2,-2,-2,0,-2,-2,0],
      [-1,-3,-2,0,0,-2,0,0,-3,-3,-3,-3,150,-3,0,0,0,0,-2,200,0],
      [800,-3,-2,-3,450,-2,0,-3,-3,200,-1,250,-1,-3,0,-1,-1,0,0,0,0],
    ]
    map2 = [
      [200,-2,-2,250,-2,0,-2,-1,0,-3,500,-3,-3,0,0,0,0,0,-3,450,-3],
      [-2,-2,-1,-2,-1,0,-3,200,-2,0,-3,0,150,0,-2,-1,0,0,0,-3,0],
      [-3,-2,-1,-3,-1,0,0,-3,-2,0,0,0,-1,-2,450,-2,0,-2,150,-2,0],
      [300,-3,-3,300,-2,-2,-2,-2,300,-2,-2,0,0,0,-2,0,-3,-3,-3,-2,0],
      [-3,-3,0,-3,-1,350,-1,0,-2,-2,350,-2,0,-3,0,0,-3,300,-3,-2,250],
      [-3,0,0,0,-1,-1,-1,-1,-3,0,-2,0,-3,400,-3,0,-3,-3,-3,0,0],
      [450,-3,0,0,0,0,-1,400,-3,0,0,0,-2,-3,-2,0,0,0,0,0,0],
      [-3,0,-1,0,-1,0,-3,-3,-3,0,-1,0,0,0,250,-3,-3,-3,-1,-2,-2],
      [0,0,-1,200,-1,-3,500,-3,0,-1,200,-1,0,0,-2,-2,-2,-1,400,-1,-2],
    ]
    map3 = [
      [0,-1,0,0,0,0,0,-3,0,-1,0,-1,-3,150,500,200,-1,0,-2,-1,0],
      [-3,500,-3,-2,350,-1,0,0,0,0,0,0,-1,-1,50,-1,0,0,-2,350,-3],
      [-1,-3,-2,0,-1,-2,0,-1,0,0,0,0,0,0,-2,-1,0,0,-2,-1,-3],
      [0,-3,0,0,0,0,-2,600,-3,0,0,-2,-2,0,-2,0,0,0,0,0,0],
      [0,0,0,0,-1,-1,0,-3,0,-1,400,-3,-2,0,0,0,-1,-2,-1,-1,-1],
      [-1,0,0,-1,-3,-3,-1,-1,0,0,-3,-2,0,0,0,0,-1,700,-1,-1,-3],
      [350,-2,-1,-3,-2,-3,-3,-1,0,-3,0,0,0,0,-3,0,-1,-1,-1,-3,200],
      [0,-1,-3,-2,250,-2,-3,-1,0,0,0,0,-2,0,0,0,0,-1,-3,300,0],
      [0,-3,-2,300,1000,-2,-3,-1,0,-1,0,-2,100,-2,-1,0,-1,-3,400,0,800],
    ]
    map4 = [
      [0,0,0,0,0,0,0,-2,0,0,0,0,-2,0,0,0,0,-1,-1,800,-1],
      [100,-1,-3,-2,0,0,-1,200,-1,-2,250,-2,-2,250,-1,0,0,-1,-2,-1,-2],
      [-1,700,-1,-3,150,-1,-1,-3,-3,-2,-2,-1,-3,-2,-1,0,0,-3,0,-1,0],
      [0,-1,-3,0,-1,0,-3,-3,-1,-3,-1,600,-1,-3,-2,-1,-3,500,-1,0,0],
      [0,-3,0,0,0,0,-2,-1,350,-1,50,-1,-3,50,0,0,0,-1,0,0,-2],
      [0,0,0,-3,250,-3,-3,-3,-1,500,450,-1,-3,0,-1,0,0,0,0,-2,100],
      [0,-3,0,0,-3,0,0,0,-3,-1,-1,-3,0,0,-2,300,-2,-1,0,0,-2],
      [0,-1,-3,-2,0,0,0,-1,0,-3,-3,-3,-1,-2,-2,-1,-2,0,-1,0,0],
      [-1,500,-1,-3,-2,0,-1,450,-1,0,0,-1,500,-1,-2,-1,-2,0,50,0,0],
    ]
    Maps = (map0,map1,map2,map3,map4)
    return Maps


game_over_reason = (
    "playing",
    "went_out_map",
    "out_of_energy",
    "invalid_action",
    "no_more_gold",
    "no_more_step",
)


# Parameters for training a DQN model
N_EPISODES = 500
MAX_STEP = 100   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay 
MEMORY_SIZE = 100_000 #The size of the batch for storing experiences
SAVE_NETWORK = 150  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 1000 #The number of experiences are stored in the memory batch before starting replaying
INPUT_DIMS = 198 #The number of input values for the DQN model
N_ACTIONS = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map
Maps = CreateMaps()
maps = [np.array(m) for m in Maps]
HOST = "localhost"
PORT = 1111
minerEnv = MinerEnv(HOST, PORT)
minerEnv.start()



def mapID_gen():
    shuffled = np.arange(0,5)
    np.random.shuffle(shuffled)
    for i in range(len(shuffled)):
        yield shuffled[i]


N_INPUT = 27
N_ACTION = 4
agent_BTC_hint = DQN(N_INPUT, N_ACTION)
h5 = "TrainedModels/DQNmodel_20200819-1638_ep20000.h5"
agent_BTC_hint.model.load_weights(h5)
#DQNAgent.target_model.load_weights(h5)


for mapID in mapID_gen():
    try:
        #mapID = np.random.randint(0, 5)
        posID_x = np.random.randint(MAP_MAX_X) 
        posID_y = np.random.randint(MAP_MAX_Y)

        request = "map{},{},{},50,100".format(mapID+1, posID_x, posID_y)
        minerEnv.send_map_info(request)

        minerEnv.reset()
        s = minerEnv.get_state()
        terminate = False # This indicates whether the episode has ended
        maxStep = minerEnv.state.mapInfo.maxStep

        for step in range(0, maxStep):
            gold_here = minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y)
            agent_energy = minerEnv.state.energy
            if gold_here > 0:
                if agent_energy <= 5:
                    minerEnv.step(str(constants.rest))
                else:
                    minerEnv.step(str(constants.dig))
            else:
                print(f"s.shape = {s.shape}")
                a_max = np.argmax(agent_BTC_hint.model.predict(s[np.newaxis,...]))      
                minerEnv.step(str(a_max))
            s_next = minerEnv.get_state()
            terminate = minerEnv.check_terminate()
            s = s_next
            
            if terminate == True:
                break

        print('(mapID {:d})'.format(mapID+1))
        print('(agent)   gold {: 5d}/{: 4d}   step {: 4d}   die of {}'.format(minerEnv.state.score, constants.gold_total(maps[mapID]), step+1, game_over_reason[minerEnv.state.status]))
        print("(bot1)    gold {: 5d}/{: 4d}   step {: 4d}".format(minerEnv.socket.bots[0].get_score(), constants.gold_total(maps[mapID]), minerEnv.socket.bots[0].state.stepCount))
        print("(bot2)    gold {: 5d}/{: 4d}   step {: 4d}".format(minerEnv.socket.bots[1].get_score(), constants.gold_total(maps[mapID]), minerEnv.socket.bots[1].state.stepCount))
        print("(bot3)    gold {: 5d}/{: 4d}   step {: 4d}".format(minerEnv.socket.bots[2].get_score(), constants.gold_total(maps[mapID]), minerEnv.socket.bots[2].state.stepCount))
        print()

    except Exception as e:
        import traceback
        traceback.print_exc()                
        #print("Finished.")
        break

