#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import numpy as np
import pandas as pd
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

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
tf.disable_v2_behavior()


# In[3]:


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


# In[19]:


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
        print("numberOfPlayers: ", self.userMatch.gameinfo.numberOfPlayers)

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
            print("Reset game: ", requests)
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


# In[5]:


#Bots :bot1
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

    def next_action(self):
        if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
            if self.info.energy >= 6:
                return self.ACTION_CRAFT
            else:
                return self.ACTION_FREE
        if self.info.energy < 5:
            return self.ACTION_FREE
        else:
            action = self.ACTION_GO_UP
            if self.info.posy % 2 == 0:
                if self.info.posx < self.state.mapInfo.max_x:
                    action = self.ACTION_GO_RIGHT
            else:
                if self.info.posx > 0:
                    action = self.ACTION_GO_LEFT
                else:
                    action = self.ACTION_GO_DOWN
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


# In[6]:


#Bots :bot2
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

    def next_action(self):
        if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
            if self.info.energy >= 6:
                return self.ACTION_CRAFT
            else:
                return self.ACTION_FREE
        if self.info.energy < 5:
            return self.ACTION_FREE
        else:
            action = np.random.randint(0, 4)            
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


# In[7]:


#Bots :bot3
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


# In[8]:


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


# In[9]:


#MinerEnv.py
TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self):
        self.socket = GameSocket()
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

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)

        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        #Add position of bots 
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action
            
        #If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= TreeID
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= TrapID
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= SwampID

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -10
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -10
        # print ("reward",reward)
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING


# In[10]:


#DQNModel.py
class DQN: 
    def __init__(self,
                 input_dim, #The number of inputs for the DQN network
                 action_space, #The number of actions for the DQN network
                 gamma = 0.99, #The discount factor
                 epsilon = 1, #Epsilon - the exploration factor
                 epsilon_min = 0.01, #The minimum epsilon 
                 epsilon_decay = 0.999,#The decay epislon for each update_epsilon time
                 learning_rate = 0.00025, #The learning rate for the DQN network
                 tau = 0.125, #The factor for updating the DQN target network from the DQN network
                 model = None, #The DQN model
                 target_model = None, #The DQN target model 
                 sess=None):

        self.input_dim = input_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
              
        #Creating networks
        self.model        = self.create_model() #Creating the DQN model
        self.target_model = self.create_model() #Creating the DQN target model
        
        #Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(sess)
        self.sess.run(tf.global_variables_initializer()) 
      
    def create_model(self):
      model = Sequential()
      model.add(Dense(300, input_dim=self.input_dim))
      model.add(Activation('relu'))
      model.add(Dense(300))
      model.add(Activation('relu'))
      model.add(Dense(self.action_space))
      # (?1) "linear" activation, isn't it equiv. to no activation at all?
      model.add(Activation('linear'))    
      #adam = optimizers.adam(lr=self.learning_rate)
      sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
      model.compile(optimizer=sgd, loss='mse')
      return model
  
    
    def act(self,state):
      #Get the index of the maximum Q values      
      a_max = np.argmax(self.model.predict(state.reshape(1,len(state))))      
      if (random.random() < self.epsilon):
        a_chosen = randrange(self.action_space)
      else:
        a_chosen = a_max      
      return a_chosen
    
    
    def replay(self,samples,batch_size):
      inputs = np.zeros((batch_size, self.input_dim))
      targets = np.zeros((batch_size, self.action_space))
      
      for i in range(0,batch_size):
        state = samples[0][i,:]
        action = samples[1][i]
        reward = samples[2][i]
        new_state = samples[3][i,:]
        done= samples[4][i]
        
        inputs[i,:] = state
        targets[i,:] = self.target_model.predict(state.reshape(1,len(state)))        
        if done:
          targets[i,action] = reward # if terminated, only equals reward
        else:
          Q_future = np.max(self.target_model.predict(new_state.reshape(1,len(new_state))))
          targets[i,action] = reward + Q_future * self.gamma
      #Training
      loss = self.model.train_on_batch(inputs, targets)  

    def target_train(self): 
      weights = self.model.get_weights()
      target_weights = self.target_model.get_weights()
      for i in range(0, len(target_weights)):
        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

      self.target_model.set_weights(target_weights) 


    def update_epsilon(self):
        self.epsilon =  self.epsilon*self.epsilon_decay
        self.epsilon =  max(self.epsilon_min, self.epsilon)


    def save_model(self, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(model_name + ".h5")
            print("Saved model to disk")


# In[11]:


#Memory.py
class Memory:        
    capacity = None  
    
    def __init__(
            self,
            capacity,
            length = None,
            states = None,
            actions = None,
            rewards = None,
            dones = None,
            states2 = None,       
    ):
        self.capacity = capacity
        self.length = 0
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.states2 = states2

    def push(self, s, a, r, done, s2):
        if self.states is None:
            self.states = s
            self.actions = a
            self.rewards = r
            self.dones = done
            self.states2 = s2
        else:
            self.states = np.vstack((self.states,s))
            self.actions = np.vstack((self.actions,a))
            self.rewards = np.vstack((self.rewards, r))
            self.dones = np.vstack((self.dones, done))
            self.states2 = np.vstack((self.states2,s2))
        
        self.length = self.length + 1
            
        if (self.length > self.capacity): 
            self.states = np.delete(self.states,(0), axis = 0)
            self.actions = np.delete(self.actions,(0), axis = 0)
            self.rewards = np.delete(self.rewards,(0), axis = 0)
            self.dones = np.delete(self.dones,(0), axis = 0)
            self.states2 = np.delete(self.states2,(0), axis = 0)           
            self.length = self.length - 1
            
        
    def sample(self,batch_size):
        if (self.length >= batch_size):
            idx = random.sample(range(0,self.length),batch_size)
            s = self.states[idx,:]
            a = self.actions[idx,:]
            r = self.rewards[idx,:]
            d = self.dones[idx,:]
            s2 = self.states2[idx,:]
                
            return list([s,a,r,s2,d])


# In[12]:


#Creating Maps
#This function is used to create 05 maps instead of loading them from Maps folder in the local
def CreateMaps():
      map1 = [
        [0,  0,  -2,  100,  0,  0,  -1,  -1,  -3,  0,  0,  0,  -1,  -1,  0,  0,  -3,  0,  -1,  -1,0],
        [-1,-1,  -2,  0, 0,  0,  -3,  -1,  0,  -2,  0,  0,  0,  -1,  0,  -1,  0,  -2,  -1,  0,0],
        [0,  0,  -1,  0,  0,  0,  0,  -1,  -1,  -1,  0, 0,  100,  0,  0,  0,  0,  50,  -2,  0,0],
        [0,  0,  0,  0,  -2,  0,  0,  0,  0,  0,  0,  0,  -1,  50, -2,  0,  0,  -1,  -1,  0,0],
        [-2, 0,  200,  -2,  -2,  300,  0, 0,  -2,  -2,  0,  0,  -3,  0,  -1,  0,  0,  -3,  -1,  0,0],
        [0,  -1,  0,  0,  0,  0,  0,  -3,  0,  0,  -1,  -1,  0,  0,  0,  0,  0,  0,  -2,  0,0],
        [0,  -1,  -1,  0,  0,  -1,  -1,  0,  0,  700,  -1,  0,  0,  0,  -2,  -1,  -1,  0,  0, 0,100],
        [0,  0, 0, 500,  0,  0,  -1,  0,  -2,  -2,  -1,  -1,  0,  0,  -2,  0,  -3,  0,  0,  -1,0],
        [-1,  -1, 0,-2 ,  0,  -1,  -2,  0,  400,  -2,  -1,  -1,  500,  0,  -2,  0,  -3,  100,  0, 0,0]
      ]
      map2 = [
        [0,  0,  -2,  0,  0,  0,  -1,  -1,  -3,  0,  0,  0,  -1,  -1,  0,  0,  -3,  0,  -1,  -1,0],
        [-1,-1,  -2,  100, 0,  0,  -3,  -1,  0,  -2,  100,  0,  0,  -1,  0,  -1,  0,  -2,  -1,  0,0],
        [0,  0,  -1,  0,  0,  0,  0,  -1,  -1,  -1,  0, 0,  0,  0,  0,  0,  50,  0,  -2,  0,0],
        [0,  200,  0,  0,  -2,  0,  0,  0,  0,  0,  0,  0,  -1,  50, -2,  0,  0,  -1,  -1,  0,0],
        [-2, 0,  0,  -2,  -2,  0,  0, 0,  -2,  -2,  0,  0,  -3,  0,  -1,  0,  0,  -3,  -1,  0,0],
        [0,  -1,  0,  0,  300,  0,  0,  -3,  0,  0,  -1,  -1,  0,  0,  0,  0,  0,  0,  -2,  0,0],
        [500,  -1,  -1,  0,  0,  -1,  -1,  0,  700,  0,  -1,  0,  0,  0,  -2,  -1,  -1,  0,  0, 0,0],
        [0,  0, 0, 0,  0,  0,  -1,  0,  -2,  -2,  -1,  -1,  0,  0,  -2,  0,  -3,  100,  0,  -1,0],
        [-1,  -1, 0,-2 ,  0,  -1,  -2,  400,  0,  -2,  -1,  -1,  0,  500,  -2,  0,  -3,  0,  0, 100,0]
      ]
      map3= [
        [0,  0,  -2,  0,  0,  0,  -1,  -1,  -3,  0,  100,  0,  -1,  -1,  0,  0,  -3,  0,  -1,  -1,0],
        [-1,-1,  -2,  0, 0,  0,  -3,  -1,  0,  -2,  0,  0,  0,  -1,  0,  -1,  0,  -2,  -1,  0,0 ],
        [0,  0,  -1,  0,  0,  0,  100,  -1,  -1,  -1,  0, 0,  50,  0,  0,  0,  50,  0,  -2,  0,0],
        [0,  200,  0,  0,  -2,  0,  0,  0,  0,  0,  0,  0,  -1,  0, -2,  0,  0,  -1,  -1,  0,0],
        [-2, 0,  0,  -2,  -2,  0,  0, 0,  -2,  -2,  0,  0,  -3,  0,  -1,  0,  0,  -3,  -1,  0,0],
        [0,  -1,  0, 300,  0,  0,  0,  -3,  0,  0,  -1,  -1,  0,  0,  0,  0,  0,  0,  -2,  0,0],
        [0,  -1,  -1,  0,  0,  -1,  -1,  700,  0,  0,  -1,  0,  0,  0,  -2,  -1,  -1,  0,  0, 0,0],
        [0,  0, 0, 0,  0,  500,  -1,  0,  -2,  -2,  -1,  -1,  0,  0,  -2,  0,  -3,  0,  700,  -1,0],
        [-1,  -1, 0,-2 ,  0,  -1,  -2,  400,  0,  -2,  -1,  -1,  0,  500,  -2,  0,  -3,  0,  0, 100,0]
      ]
      map4=[
        [0,  0,  -2,  0,  0,  0,  -1,  -1,  -3,  0,  0,  0,  -1,  -1,  0,  0,  -3,  0,  -1,  -1,0],
        [-1,-1,  -2,  0, 0,  0,  -3,  -1,  0,  -2,  0,  0,  100,  -1,  0,  -1,  0,  -2,  -1,  0,0],
        [0,  0,  -1,  0,  100,  0,  0,  -1,  -1,  -1,  0, 0,  0,  0,  50,  0,  50,  0,  -2,  0,0],
        [0,  200,  0,  0,  -2,  0,  0,  0,  0,  0,  0,  0,  -1,  0, -2,  0,  0,  -1,  -1,  0,0],
        [-2, 0,  0,  -2,  -2,  0,  0, 0,  -2,  -2,  0,  0,  -3,  0,  -1,  0,  0,  -3,  -1,  0,0],
        [0,  -1,  0,  0,  0,  0,  300,  -3,  0,  700,  -1,  -1,  0,  0,  0,  0,  0,  0,  -2,  0,0],
        [0,  -1,  -1,  0,  0,  -1,  -1,  0,  0,  0,  -1,  0,  0,  0,  -2,  -1,  -1,  0,  0, 100,0],
        [500,  0, 0, 0,  0,  0,  -1,  0,  -2,  -2,  -1,  -1,  0,  0,  -2,  0,  -3,  0,  0,  -1,0],
        [-1,  -1, 0,-2 ,  0,  -1,  -2,  400,  0,  -2,  -1,  -1,  0,  500,  -2,  0,  -3,  0,  0, 100,0]

      ]
      map5=[
        [0,  0,  -2,  0,  100,  0,  -1,  -1,  -3,  0,  0,  0,  -1,  -1,  0,  0,  -3,  0,  -1,  -1,0],
        [-1,-1,  -2,  0, 0,  0,  -3,  -1,  0,  -2,  100,  0,  0,  -1,  0,  -1,  0,  -2,  -1,  0,0],
        [0,  0,  -1,  0,  0,  0,  0,  -1,  -1,  -1,  0, 0,  0,  0,  50,  0,  0,  0,  -2,  0,0],
        [0,  200,  0,  0,  -2,  0,  0,  0,  0,  0,  0,  0,  -1,  0, -2,  0,  50,  -1,  -1,  0,0],
        [-2, 0,  0,  -2,  -2,  0,  0, 0,  -2,  -2,  0,  0,  -3,  0,  -1,  0,  0,  -3,  -1,  0,0],
        [0,  -1,  0,  0,  300,  0,  0,  -3,  0,  0,  -1,  -1,  0,  0,  0,  0,  0,  0,  -2,  0,0],
        [500,  -1,  -1,  0,  0,  -1,  -1,  0,  0,  700,  -1,  0,  0,  0,  -2,  -1,  -1,  0,  0, 100,0],
        [0,  0, 0, 0,  0,  0,  -1,  0,  -2,  -2,  -1,  -1,  0,  0,  -2,  0,  -3,  0,  0,  -1,0],
        [-1,  -1, 0,-2 ,  0,  -1,  -2,  400,  0,  -2,  -1,  -1,  0,  500,  -2,  0,  -3,  0,  0, 100,0]
      ]
      Maps = (map1,map2,map3,map4,map5)
      return Maps   


# In[ ]:


#DQN Algorithm-Main
#Create header for saving DQN learning file
'''now = datetime.datetime.now()
header = ["Ep","Step", "Reward","Total_reward","Action","Epsilon","Done","Termination_Code"]
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns = header).to_csv(f,encoding='utf-8', index=False, header = True)'''

# Parameters for training a DQN model
N_EPISODES = 10_000
MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay 
MEMORY_SIZE = 100_000 #The size of the batch for storing experiences
SAVE_NETWORK = 100  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 1000 #The number of experiences are stored in the memory batch before starting replaying
INPUT_DIMS = 198 #The number of input values for the DQN model
N_ACTIONS = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

# Initialize network and memory
DQNAgent = DQN(INPUT_DIMS, N_ACTIONS)
memory = Memory(MEMORY_SIZE)

# Initialize environment
Maps = CreateMaps()
minerEnv = MinerEnv() # Creating a communication environment between the DQN model and the game environment
minerEnv.start() # Connect to the game

train = False # The variable is used to indicate that the replay starts, and the epsilon starts decrease.
for episode_i in range(N_EPISODES):
    try:
        mapID = np.random.randint(0, 5)
        # Initial position of the DQN agent
        posID_x = np.random.randint(MAP_MAX_X) 
        posID_y = np.random.randint(MAP_MAX_Y)

        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
        #request = "map{},{},{},50,100".format(mapID, posID_x, posID_y)
        # (?2) 50, 100? What are these numbers?
        minerEnv.send_map_info(request)

        minerEnv.reset()
        s = minerEnv.get_state()
        total_reward = 0
        terminate = False # This indicates whether the episode has ended
        maxStep = minerEnv.state.mapInfo.maxStep # Get the maximum number of steps for each episode in training
        # Start an episde for training
        for step in range(0, maxStep):
            action = DQNAgent.act(s)  # Getting an action from the DQN model from the state (s)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state()  # Getting a new state
            reward = minerEnv.get_reward()  # Getting a reward
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode
             
            # Add this transition to the memory batch
            memory.push(s, action, reward, terminate, s_next)

            # Sample batch memory to train network
            if (memory.length > INITIAL_REPLAY_SIZE):
                #If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                #then start replaying
                batch = memory.sample(BATCH_SIZE) #Get a BATCH_SIZE experiences for replaying
                DQNAgent.replay(batch, BATCH_SIZE)#Do relaying
                train = True #Indicate the training starts
            total_reward = total_reward + reward #Plus the reward to the total rewad of the episode
            s = s_next #Assign the next state for the next step.
            
            #Saving data to file
            '''save_data = np.hstack([episode_i+1,step+1,reward,total_reward,action, DQNAgent.epsilon, terminate]).reshape(1,7)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f,encoding='utf-8', index=False, header = False)'''

            if terminate == True:
                #If the episode ends, then go to the next episode
                break
            
        # Iteration to save the network architecture and weights
        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            DQNAgent.target_train()  # Replace the learning weights for target model with soft replacement
            #Save the DQN model
            now = datetime.datetime.now() #Get the latest datetime          
            DQNAgent.save_model("DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i+1))   
        
        #Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumlated Reward = %.2f. Epsilon = %.2f .Termination code: %d' % (episode_i+1, step+1, total_reward, DQNAgent.epsilon, terminate))
        #Decreasing the epsilon if the replay starts
        if train == True:
            DQNAgent.update_epsilon()
            	
    except Exception as e:
        import traceback
        traceback.print_exc()                
        #print("Finished.")
        break

##########################
# last cell
##########################
