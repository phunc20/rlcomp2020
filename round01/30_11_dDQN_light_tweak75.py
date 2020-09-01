########################################
# Changes compared to 30_11_dDQN_light_tweak71.py
# 01. 
#   lr_optimizer = 7.3e-4
# 02. 
#   discount_rate = 0.98
# 03. 
#   max_replay_len = 30_000
########################################


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

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

import tensorflow.keras as keras

#import tensorflow.compat.v1 as tf
#from tensorflow.compat.v1.keras import backend as K
#tf.disable_v2_behavior()
import tensorflow as tf
from tensorflow.keras import backend as K

import constants
import non_RL_agent
import non_RL_agent02
import non_RL_agent03
import non_RL_agent04
import non_RL_agent05
import non_RL_agent06

n_episodes = 500_000
#n_epsilon_decay = int(n_episodes*.6)
#n_epsilon_decay = int(n_episodes*.805)
#n_epsilon_decay = 10**6 / 0.99
n_epsilon_decay = int(n_episodes // 50)
n_episodes_buf_fill = 10_000
batch_size = 32
discount_rate = 0.98
#lr_optimizer = 2.5e-4
lr_optimizer = 7.3e-4
#loss_fn = keras.losses.mean_squared_error
loss_fn = keras.losses.Huber()
max_replay_len = 30_000


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
        #return int(non_RL_agent03.greedy_policy(s))
        return int(non_RL_agent.greedy_policy(s, how_gold=non_RL_agent.find_worthiest_gold))
 
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
        return int(non_RL_agent02.greedy_policy(s))
        #if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
        #    if self.info.energy >= 6:
        #        return self.ACTION_CRAFT
        #    else:
        #        return self.ACTION_FREE
        #if self.info.energy < 5:
        #    return self.ACTION_FREE
        #else:
        #    action = self.ACTION_GO_LEFT
        #    if self.info.posx % 2 == 0:
        #        if self.info.posy < self.state.mapInfo.max_y:
        #            action = self.ACTION_GO_DOWN
        #    else:
        #        if self.info.posy > 0:
        #            action = self.ACTION_GO_UP
        #        else:
        #            action = self.ACTION_GO_RIGHT            
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
        self.energy_pre = 0
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
                self.energy_pre = self.energy
                self.energy = player["energy"]
                self.score = player["score"]
                self.lastAction = player["lastAction"]
                self.status = player["status"]

        self.mapInfo.update(new_state["golds"], new_state["changedObstacles"])
        self.players = new_state["players"]
        for i in range(len(self.players), 4, 1):
            self.players.append({"playerId": i, "posx": self.x, "posy": self.y})
        self.stepCount = self.stepCount + 1


#MinerEnv.py
TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self):
        self.socket = GameSocket()
        self.state = State()
        self.score_pre = self.state.score

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

    def get_state(self):
        """
        Fuse `view` and `energyOnMap` into a single matrix to have a simple and concise state/observation.

        We want a matrix showing the following:
        `gold`: The amount of gold
        `all the others`: The energy that each type of terrain is going to take if being stepped into, e.g.
                          `land` => -1, `trap` => -10, etc.
        """
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
        energyOnMap = np.array(self.socket.energyOnMap)

        # `view` will contribute only to the type of terrain of `gold`
        view[view <= 0] = -9999 # Just a dummy large negative number to be got rid of later
        # `energyOnMap` will contribute to the types of terrain of `land`, `trap`, `forest` and `swamp`.
        # Recall. `forest` was designated by BTC to the value of 0, to mean random integer btw [5..20].
        energyOnMap[energyOnMap == 0] = - constants.forest_energy
        channel0 = np.maximum(view, energyOnMap)
        # Finish channel 0
        # Channel 1 will contain the position of the agent
        channel1 = np.zeros_like(channel0)
        x_agent_out_of_map = self.state.x < 0 or self.state.x >= constants.width
        y_agent_out_of_map = self.state.y < 0 or self.state.y >= constants.height
        if x_agent_out_of_map or y_agent_out_of_map:
            pass
        else:
            channel1[self.state.y, self.state.x] = self.state.energy
        state = np.stack((channel0, channel1), axis=-1)
        return state


    def get_reward(self):
        # Initialize reward
        reward = 0
        if self.state.status == constants.agent_state_str2id["out_of_MAP"]:
            #if self.state.stepCount < 50:
            #    reward += -5*(50 - self.state.stepCount)
            reward -= 1000
        #elif self.state.status == constants.agent_state_str2id["no_more_STEP"]:
        #    #reward += (self.state.score/total_gold) * 100
        #    pass
        elif self.state.status == constants.agent_state_str2id["no_more_ENERGY"]:
            reward -= 300
        #elif self.state.status == constants.agent_state_str2id["no_more_GOLD"]:
        #    pass
        #elif self.state.status == constants.agent_state_str2id["INVALID_action"]:
        #    pass
        else: # Here below: we are almost sure that agent is not out of map
            s = self.get_state()
            try:
                terrain_now = s[self.state.y, self.state.x, 0]
            except Exception as e:
                print(f"{e}")
                print(f"self.state.x, self.state.y = {self.state.x}, {self.state.y} ")
                raise e
            pos_now = np.array([self.state.x, self.state.y])
            reverse_mv = constants.action_id2ndarray[constants.reverse_action_id[self.state.lastAction]]
            pos_pre = pos_now + reverse_mv
            x_pre, y_pre = pos_pre
            terrain_pre = s[y_pre, x_pre, 0]

            # Punish `dig on obstacle`
            if self.state.lastAction == constants.dig:
                if terrain_now < 0:
                    reward -= 100
                elif terrain_now > 0:
                    score_action = self.state.score - self.score_pre
                    reward += score_action
                    self.score_pre = self.state.score
            if self.state.lastAction in (constants.up, constants.down, constants.left, constants.right,): # i.e. if agent moved
                if terrain_pre > 100: # punish leaving gold
                    reward -= terrain_pre
                if terrain_now > 0: # entering gold
                    if self.state.energy > constants.punishments["gold"]:
                        reward += 50
                    else:
                        reward -= 100
                if terrain_now < 0: # punish according to terrain_now
                    reward += terrain_now
                    if terrain_now == -100: # i.e. fatal swamp
                        reward -= 500
            if self.state.lastAction == constants.rest:
                if self.state.energy_pre >= 40:
                    reward -= 200
                if self.state.energy_pre <= 5:
                    reward += 20
            if self.state.status == constants.agent_state_str2id["PLAYing"]:
                reward += 1

        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING


Maps = [constants.maps[i] for i in range(1, 6)]
env = MinerEnv() # Creating a communication environment between the DQN model and the game environment
env.start() # Connect to the game



from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


tf.random.set_seed(42)
np.random.seed(42)

#input_shape = [constants.height, constants.width, 1+4]
input_shape = [constants.height, constants.width, 1+1]
n_outputs = 6

model = keras.models.Sequential([
    Conv2D(4, 3, activation="relu", padding="same", input_shape=input_shape),
    #MaxPooling2D(2),
    Conv2D(8, 3, activation="relu", padding="same"),
    #Conv2D(128, 3, activation="relu", padding="same"),
    #MaxPooling2D(2),
    Flatten(),
    #Dense(128, activation="elu"),
    Dense(128, activation="elu"),
    Dense(64, activation="elu"),
    Dense(32, activation="elu"),
    Dense(n_outputs)
])
#h5 = "models/30_11_dDQN_light_tweak14/avg-1785.00-episode-11155-30_11_dDQN_light_tweak14-gold-1800-step-100-20200827-0903.h5"
#model = keras.models.load_model(h5)
target = keras.models.clone_model(model)
target.set_weights(model.get_weights())



from collections import deque
replay_memory = deque(maxlen=max_replay_len)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def epsilon_greedy_policy(state, epsilon=0, n_actions=6):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        #pictorial = pictorial_state(state)
        #Q_values = model.predict(pictorial[np.newaxis])
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    #next_state, reward, done, info = env.step(action)
    env.step(str(action))
    next_state = env.get_state()
    reward = env.get_reward()
    done = env.check_terminate()
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done


#optimizer = keras.optimizers.Adam(lr=1e-3)
#optimizer = keras.optimizers.Adam(lr=2.5e-4)
optimizer = keras.optimizers.Adam(lr=lr_optimizer)

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    #pictorials = np.array([pictorial_state(s) for s in states])
    #next_pictorials = np.array([pictorial_state(next_s) for next_s in next_states])
    #next_Q_values = model.predict(next_pictorials)
    next_Q_values = model.predict(next_states)
    #max_next_Q_values = np.max(next_Q_values, axis=1)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    #target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values
    target_Q_values = rewards + (1 - dones) * discount_rate * next_best_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        #all_Q_values = model(pictorials)
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


np.random.seed(42)
tf.random.set_seed(42)


from constants import n_allowed_steps

now = datetime.datetime.now()
now_str = now.strftime("%Y%m%d-%H%M")
script_name = __file__.split('.')[0]
save_path = os.path.join("models", script_name)
os.makedirs(save_path, exist_ok=True)


scores = [] 
scores_avg = [] 
best_score = 0
k = 10
scores_k_most_recent = deque([0]*k, maxlen=k)
best_score_avg = 1400

with open(os.path.join(save_path, f"log-{now_str}.txt"), 'w') as log:
    for episode in range(n_episodes):
        mapID = np.random.randint(0, 5)
        posID_x = np.random.randint(constants.width) 
        posID_y = np.random.randint(constants.height)
        request = "map{},{},{},50,100".format(mapID, posID_x, posID_y)
        env.send_map_info(request)
        env.reset()
        obs = env.get_state()
        undiscounted_return = 0
        for step in range(n_allowed_steps):
            epsilon = max(1 - episode / n_epsilon_decay, 0.01)
            obs, reward, done = play_one_step(env, obs, epsilon)
            undiscounted_return += reward
            if done:
                break
        score = env.state.score
        scores.append(score)
        scores_k_most_recent.append(score)
        #score_avg = np.mean(scores_k_most_recent)
        score_avg = round(np.mean(scores_k_most_recent), 1)
        scores_avg.append(score_avg)
        #if score > best_score:
        if score_avg > best_score_avg:
            #best_weights = model.get_weights()
            best_score_avg = score_avg 
            #best_score = score
            #model.save(os.path.join(save_path, f"episode-{episode+1}-gold-{env.state.score}-avg-{score_avg:4.2f}-step-{step+1}-{now_str}.h5"))
            model.save(os.path.join(save_path, f"avg-{score_avg:07.2f}-episode-{episode+1}-{__file__.split('.')[0]}-gold-{env.state.score}-step-{step+1}-{now_str}.h5"))
    
        #message = "(Episode {: 5d}/{})   Gold {: 4d}  avg {: 8.2f}  undisc_return {: 6d}   step {: 3d}   eps {:.2f}  ({})\n".format(episode+1, n_episodes, env.state.score, score_avg, undiscounted_return, step + 1, epsilon, constants.agent_state_id2str[env.state.status])
        message = "(Episode {: 5d}/{})   Gold {: 4d}  avg {: 8.1f}  undisc_return {: 6d}   step {: 3d}   eps: {:.2f}  ({})\n".format(episode+1, n_episodes, env.state.score, score_avg, undiscounted_return, step + 1, epsilon, constants.agent_state_id2str[env.state.status])
        ##############################################
        #score = env.state.score*(n_allowed_steps - step)
        #score = env.state.score
        #scores.append(score)
        #if score > best_score:
        #    #best_weights = model.get_weights()
        #    best_score = score
        #    model.save(os.path.join(save_path, f"episode-{episode+1}-gold-{env.state.score}-step-{step+1}-{now_str}.h5"))
    
        #message = "(Episode {: 5d}/{})   Gold: {: 4d}  undiscounted_return: {: 6d}   Steps: {: 3d}   eps: {:.3f}  ({})\n".format(episode+1, n_episodes, env.state.score, undiscounted_return, step + 1, epsilon, constants.agent_state_id2str[env.state.status])
        print(message, end='')
        log.write(message)
    
        #if episode > 500:
        if episode > n_episodes_buf_fill:
            training_step(batch_size)
        if episode % n_episodes_buf_fill == 0:
            target.set_weights(model.get_weights())

#np.save(f"scores-{now_str}", np.array(scores))
#np.save(f"scores-N-scores_avg-{now_str}", np.array([scores, scores_avg]))
np.save(f"scores-N-scores_avg-{__file__.split('.')[0]}-{now_str}", np.array([scores, scores_avg]))
