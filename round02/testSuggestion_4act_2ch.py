"""
Diff from testSuggestion_01_6act.py:

When agent suggests a mv that will zero out its energy,
testSuggestion_02_6act.py, i.e. the current script,
will choose to take a rest, and when preceding to the next step,
will let the DQN model suggest again which, amongst the 6 actions,
would be the most recommended.

"""


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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import optimizers

import tensorflow.keras as keras

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
tf.disable_v2_behavior()

import constants02
from constants02 import width, height, forest_energy, action_id2ndarray, action_id2str
import non_RL_agent
import non_RL_agent02
import non_RL_agent03
import non_RL_agent04
import non_RL_agent05
import non_RL_agent06
from miner_env import MinerEnv

def mapID_gen():
    shuffled = np.arange(0,5)
    np.random.shuffle(shuffled)
    for i in range(len(shuffled)):
        yield shuffled[i]

maps = [np.array(constants02.maps[i]) for i in range(1, 6)]
env = MinerEnv()
env.start()

final_score = 0
bot1_final_score = 0
bot2_final_score = 0
bot3_final_score = 0
#h5 = "models/tf2.3.0/38_02_dDQN_4act/avgGold-1890.0-episode-219913-38_02_dDQN_4act-visitRatio-0.25-gold-0-step-15-20200829-2359.h5"
#h5 = "models/tf2.3.0/38_02_dDQN_4act_fromZero/avgGold-1560.0-episode-269135-38_02_dDQN_4act_fromZero-visitRatio-0.18181818181818182-gold-0-step-11-20200830-0001.h5"
h5 = "03_4act/models/00_4act_2channel/avg-1405.00-episode-317121-00_4act_2channel-visitRatio-0.10-gold-0-step-9-20200913-0113.h5"
agent = keras.models.load_model(h5)
#input_shape = [constants02.height, constants02.width, 1+1]
#n_outputs = 4
#agent = keras.models.Sequential([
#    Conv2D(4, 3, activation="relu", padding="same", input_shape=input_shape),
#    Conv2D(8, 3, activation="relu", padding="same"),
#    Flatten(),
#    Dense(128, activation="elu"),
#    Dense(64, activation="elu"),
#    Dense(32, activation="elu"),
#    Dense(n_outputs)
#])

for mapID in mapID_gen():
    try:
        #mapID = np.random.randint(0, 5)
        posID_x = np.random.randint(width) 
        posID_y = np.random.randint(height)

        request = "map{},{},{},50,100".format(mapID+1, posID_x, posID_y)
        env.send_map_info(request)

        env.reset()
        s = env.get_view_9x21x5()[...,:2]
        terminate = False # This indicates whether the episode has ended
        maxStep = env.state.mapInfo.maxStep
        #logging.debug(f"maxStep = {maxStep}")
        #print(f"maxStep = {maxStep}")
        for step in range(0, maxStep):
            ## Non-RL
            #env.step(non_RL_agent.greedy_policy(s))
            #env.step(non_RL_agent.greedy_policy(s, how_gold=non_RL_agent.find_worthiest_gold))
            #env.step(non_RL_agent04.greedy_policy(env, how_gold=non_RL_agent.find_worthiest_gold))
            #env.step(non_RL_agent05.greedy_policy(env, how_gold=non_RL_agent.find_worthiest_gold))
            #env.step(non_RL_agent05.greedy_policy(env))
            #env.step(non_RL_agent06.greedy_policy(s))

            ## 4-action RL agent
            #view = env.get_state()
            view = env.view_9x21x5[..., :2]
            x_agent = env.state.x
            y_agent = env.state.y
            pos_agent = np.array([x_agent, y_agent])
            energy_agent = env.state.energy
            suggested_Qs = agent.predict(s[np.newaxis,...])[0]
            suggested_action_ids = np.argsort(suggested_Qs)
            #print("new step")
            while True:
                # 01) on Gold and enough Energy => dig
                # 02) on Gold and NOT enough Energy => rest
                # 03) on non-Gold => refer to agent's suggestion
                if view[y_agent, x_agent, 0] > 0: # i.e. if standing on gold
                    if energy_agent > 5:
                        most_suggested_action_id = constants02.dig
                    else:
                        most_suggested_action_id = constants02.rest
                    break
                else: # i.e. NOT standing on gold
                    if len(suggested_action_ids) == 0:
                        ## N.B. we randomly choose a non-dig action in this case,
                        ## i.e. randint(0,5) instead of randint(0,6) because 6 equals `dig`
                        #most_suggested_action_id = np.random.randint(0,5)
                        most_suggested_action_id = constants02.rest
                        print("pseudo-random step")
                        break
                    most_suggested_action_id = suggested_action_ids[-1]
                    #if most_suggested_action_id == constants02.rest: # i.e. if agent suggests `rest`
                    #    if energy_agent == constants02.max_energy: # worthless rest
                    #        suggested_action_ids = np.delete(suggested_action_ids, -1)
                    #    else:
                    #        break
                    most_suggested_mv = action_id2ndarray[most_suggested_action_id]
                    pos_mv = pos_agent + most_suggested_mv
                    x_mv, y_mv = pos_mv
                    #print(f"suggested_actions = {[ constants02.action_id2str[id_] for id_ in suggested_action_ids]}")
                    # TODO
                    if constants02.out_of_map(pos_mv):
                        #print("stopped out map")
                        suggested_action_ids = np.delete(suggested_action_ids, -1)
                    elif view[y_mv, x_mv, 0] > 0 and energy_agent <= 4:
                        most_suggested_action_id = constants02.rest
                        break
                    elif view[y_mv, x_mv, 0] <= -constants02.max_energy:
                        suggested_action_ids = np.delete(suggested_action_ids, -1)
                    elif energy_agent + view[y_mv, x_mv, 0] <= 0:
                        most_suggested_action_id = constants02.rest
                        break
                    else:
                        #print(f"pos_agent = {pos_agent}")
                        #print(f"energy_agent = {energy_agent}")
                        #print(f"pos_mv = {pos_mv}")
                        #print(f"view[y_mv, x_mv, 0] = {view[y_mv, x_mv, 0]}")
                        #print(f"action = {action_id2str[most_suggested_action_id]}")
                        break

            #env.step(str(a_max))
            env.step(str(most_suggested_action_id))
            #s_next = env.get_state()
            s_next = env.get_view_9x21x5()[...,:2]
            terminate = env.check_terminate()
            s = s_next
            
            if terminate == True:
                #print(f"pos_agent = {pos_agent}")
                #print(f"energy_agent = {energy_agent}")
                #print(f"pos_mv = {pos_mv}")
                #print(f"most_suggested_mv = {most_suggested_mv}")
                try:
                    #print(f"view[y_mv, x_mv, 0] = {view[y_mv, x_mv, 0]}")
                    pass
                except IndexError:
                    #print(f"x_mv, y_mv = {x_mv}, {y_mv}")
                    pass
                #print(f"action = {action_id2str[most_suggested_action_id]}")
                break

        print('(mapID {:d})'.format(mapID+1))
        print('(agent)   gold {: 5d}/{: 4d}   step {: 4d}   energy {: 2d}   ({})'.format(env.state.score, constants02.gold_total(maps[mapID]), step+1, env.state.energy, constants02.agent_state_id2str[env.state.status]))
        #print("(bot1)    gold {: 5d}/{: 4d}   step {: 4d}".format(env.socket.bots[0].get_score(), constants02.gold_total(constants02.maps[mapID]), env.socket.bots[0].state.stepCount))
        #print("(bot2)    gold {: 5d}/{: 4d}   step {: 4d}".format(env.socket.bots[1].get_score(), constants02.gold_total(constants02.maps[mapID]), env.socket.bots[1].state.stepCount))
        #print("(bot3)    gold {: 5d}/{: 4d}   step {: 4d}".format(env.socket.bots[2].get_score(), constants02.gold_total(constants02.maps[mapID]), env.socket.bots[2].state.stepCount))
        #bot1_final_score += env.socket.bots[0].get_score()
        #bot2_final_score += env.socket.bots[1].get_score()
        #bot3_final_score += env.socket.bots[2].get_score()


        bot1_score = [player["score"] for player in env.socket.bots[1].state.players if player["playerId"] == 2][0]
        bot2_score = [player["score"] for player in env.socket.bots[1].state.players if player["playerId"] == 3][0]
        bot3_score = [player["score"] for player in env.socket.bots[1].state.players if player["playerId"] == 4][0]
        print("(bot1)    gold {: 5d}/{: 4d}   step {: 4d}".format(bot1_score, constants02.gold_total(maps[mapID-1]), env.socket.bots[0].state.stepCount))
        print("(bot2)    gold {: 5d}/{: 4d}   step {: 4d}".format(bot2_score, constants02.gold_total(maps[mapID-1]), env.socket.bots[1].state.stepCount))
        print("(bot3)    gold {: 5d}/{: 4d}   step {: 4d}".format(bot3_score, constants02.gold_total(maps[mapID-1]), env.socket.bots[2].state.stepCount))
        print()
        final_score += env.state.score
        bot1_final_score += bot1_score
        bot2_final_score += bot2_score
        bot3_final_score += bot3_score



    except Exception as e:
        import traceback
        traceback.print_exc()                
        #print("Finished.")
        break

print(f"final_score      = {final_score}")
print(f"bot1_final_score = {bot1_final_score}")
print(f"bot2_final_score = {bot2_final_score}")
print(f"bot3_final_score = {bot3_final_score}")
