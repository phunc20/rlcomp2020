import numpy as np
import datetime
import json
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)

import constants02
#import non_RL_agent
#import non_RL_agent02
#import non_RL_agent03
#import non_RL_agent04
#import non_RL_agent05
#import non_RL_agent06
import revived_non_RL_01
from miner_env import MinerEnv

#maps = [np.array(m) for m in Maps]
maps = [np.array(constants02.maps[i]) for i in range(1, 6)]
env = MinerEnv()
env.start()

goat = revived_non_RL_01.Goat()

def mapID_gen():
    #shuffled = np.arange(0,5)
    shuffled = np.arange(1,6)
    np.random.shuffle(shuffled)
    for i in range(len(shuffled)):
        yield shuffled[i]

final_score = 0
bot1_final_score = 0
bot2_final_score = 0
bot3_final_score = 0

for mapID in mapID_gen():
    try:
        #mapID = np.random.randint(0, 5)
        posID_x = np.random.randint(constants02.width)
        posID_y = np.random.randint(constants02.height)

        request = "map{},{},{},50,100".format(mapID, posID_x, posID_y)
        env.send_map_info(request)

        env.reset()
        #print('(mapID {:d})'.format(mapID))
        #view, energy, stepCount, pos_players = env.get_non_RL_state_01()
        view, energy, stepCount, pos_players = env.get_non_RL_state_02()
        #print(f"view.shape = {view.shape}")
        terminate = False
        #maxStep = env.state.mapInfo.maxStep
        #logging.debug(f"maxStep = {maxStep}")
        #print(f"maxStep = {maxStep}")
        #try:
        #    logging.debug(f"energy = {env.state.energy}, pos = ({env.state.x:2d},{env.state.y:2d}), target = {goat.pos_target_gold}({goat.view[goat.pos_target_gold[1], goat.pos_target_gold[0]]}), here = {view[env.state.y, env.state.x]}, score = {env.state.score}")
        #except:
        #    logging.debug(f"energy = {env.state.energy}, pos = ({env.state.x:2d},{env.state.y:2d}), target = {goat.pos_target_gold}, here = {view[env.state.y, env.state.x]}, score = {env.state.score}")
        for step in range(constants02.n_allowed_steps):
            logging.debug(f"\n\n")
            #logging.debug(f"step = {step + 1}")
            #print(f"##########################")
            #print(f"## step = {step + 1}")
            #print(f"##########################")
            logging.debug(f"##########################")
            logging.debug(f"## step = {step + 1}")
            logging.debug(f"##########################")
            ## non-RL
            str_ = str(goat.policy_nearest_gold(view, energy, pos_players))
            #str_ = str(goat.policy_00(env))
            env.step(str_)
            #if not goat.pos_target_gold is None:
            #    #print(f"energy = {env.state.energy}, pos = ({env.state.x:2d},{env.state.y:2d}), lastAction = {constants02.action_id2str[env.state.lastAction]}, target = {goat.pos_target_gold}, amount = {view[goat.pos_target_gold[1], goat.pos_target_gold[0]]}, score = {env.state.score}")
            #    logging.debug(f"energy = {env.state.energy}, pos = ({env.state.x:2d},{env.state.y:2d}), lastAction = {constants02.action_id2str[env.state.lastAction]}, target = {goat.pos_target_gold}({goat.view[goat.pos_target_gold[1], goat.pos_target_gold[0]]}), here = {view[env.state.y, env.state.x]}, score = {env.state.score}")
            #else:
            #    #print(f"energy = {env.state.energy}, pos = ({env.state.x:2d},{env.state.y:2d}), lastAction = {constants02.action_id2str[env.state.lastAction]}, target = {goat.pos_target_gold}, score = {env.state.score}")
            #    logging.debug(f"energy = {env.state.energy}, pos = ({env.state.x:2d},{env.state.y:2d}), lastAction = {constants02.action_id2str[env.state.lastAction]}, target = {goat.pos_target_gold}, score = {env.state.score}")
            #env.step(str(constants02.rest))
            #view, energy, stepCount, pos_players = env.get_non_RL_state_01()
            view, energy, stepCount, pos_players = env.get_non_RL_state_02()
            terminate = env.check_terminate()
            
            if terminate == True:
                break

        print('(mapID {:d})'.format(mapID))
        print('(agent)   gold {: 5d}/{: 4d}   step {: 4d}   die of {}'.format(env.state.score, constants02.gold_total(maps[mapID-1]), step+1, constants02.agent_state_id2str[env.state.status]))
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
