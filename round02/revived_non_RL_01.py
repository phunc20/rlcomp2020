"""
methods' name starting with
01) find_: are older/deprecated methods
02) trouver_: are newer methods which usually have less input args


"""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter
#import constants02
from constants02 import *
#from viz_utils02 import *
from collections import deque
#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

class Goat:
    """
    I wanted to name this class Gott, but too impolite. So... Goat.
    """
    def __init__(self, n_lingering=100):
        self.pos_impossible_golds = []
        self.pos_possible_golds = []
        self.pos_target_gold = None
        self.distance_to_target = deque(maxlen=n_lingering)
        self.view = None
        self.pos = None
        self.x, self.y = None, None
        self.energy = None
        self.stepCount = None
        self.n_remain_steps = None
        self.pos_other_players = None

    def valeur(self, pos_gold, n_approx_steps, k=2):
        #propre_valeur = 
        #extra_valeur = 
        # TODO: Take swamp into consideration
        valeurs = []
        poids = []
        for pos_vang in self.pos_possible_golds:
            distance = l1_dist(pos_vang, pos_gold)
            if distance > k:
                continue
            else:
                poids.append(2**(-k))
                valeurs.append(self.view[pos_vang[1], pos_vang[0]])
        valeurs = np.array(valeurs)
        poids = np.array(poids)
        total = max(0, (valeurs*poids).sum() - n_approx_steps*15)
        n_share = 1
        for pos_opponent in self.pos_other_players:
            if l1_dist(pos_opponent, pos_gold) <= k:
                n_share += 1
        total = total / 4
        return total

    #def evaluate_00(pos_gold, n_remain_steps):
    #def trouver_most_valuable_gold(pos_gold, n_remain_steps):
    def trouver_most_valuable_gold(self, pos_from=self.pos, ratio=2/3):
        # TODO
        n_remain_mv = round(self.n_remain_steps * ratio) # we assume every 3 steps need 1 rest, i.e. 2/3 of the time spending on real movement
        values = []
        duppleganger = self.pos_possible_golds.copy()
        self.pos_possible_golds = []
        for pos_gold in duppleganger:
            distance = l1_dist(pos_gold, pos_from)
            n_approx_needed_steps = round(distance / ratio)
            #if distance >= n_remain_mv:
            if n_approx_needed_steps >= self.n_remain_steps:
                self.pos_impossible_golds.append(pos_gold)
                #self.update_pos_possible_golds(pos_gold)
                #self.pos_possible_golds.append(pos_gold)
            else:
                ## TODO: n_left_steps to be tuned
                n_left_steps = max(0, self.n_remain_steps - n_approx_needed_steps - 1)
                value = self.valeur(pos_gold, n_left_steps)
                values.append(value)
                self.pos_possible_golds.append(pos_gold)
        values = np.array(values)
        self.pos_possible_golds = np.array(self.pos_possible_golds, dtype=np.int8)
        #index = np.argmax(values, axis=)
        index = np.argmax(values)
        return self.pos_possible_golds[index]

    def find_pos_golds(self, view):
        row_col_golds = np.argwhere(view>0)
        pos_golds = np.zeros_like(row_col_golds)
        pos_golds[:,0], pos_golds[:,1] = row_col_golds[:,1], row_col_golds[:,0]
        return pos_golds

    def trouver_pos_golds(self):
        row_col_golds = np.argwhere(self.view>0)
        pos_golds = np.zeros_like(row_col_golds)
        pos_golds[:,0], pos_golds[:,1] = row_col_golds[:,1], row_col_golds[:,0]
        return pos_golds

    def one_step_closer_to(self):
        pass

    def towards_target(self):
        # TODO: recursive to itself when too long to prev. target
        #if len(self.distance_to_target) > too_long:
        #    logging.warning("too_long happens")
        #    self.pos_impossible_golds.append(self.pos_target_gold)
        #elif len(self.distance_to_target) > long_ and self.distance_to_target[-1] >= min(list(self.distance_to_target)[:-1]):
        #    self.pos_impossible_golds.append(self.pos_target_gold)
        self.distance_to_target.append(l1_dist(self.pos_target_gold, self.pos))
        ## Try to follow the shortest path to target
        needed_displacements = []
        #vec_displacement = pos_nearest_gold - pos_agent
        vec_displacement = self.pos_target_gold - self.pos
        # horizontal, i.e. x-movement
        if vec_displacement[0] > 0:
            #needed_displacements.extend(["right"]*vec_displacement[0])
            needed_displacements.append(right)
        elif vec_displacement[0] < 0:
            #needed_displacements.extend(["left"]*vec_displacement[0])
            needed_displacements.append(left)
        
        # vertical, i.e. y-movement
        if vec_displacement[1] > 0:
            #needed_displacements.extend(["down"]*vec_displacement[0])
            needed_displacements.append(down)
        elif vec_displacement[1] < 0:
            #needed_displacements.extend(["up"]*vec_displacement[0])
            needed_displacements.append(up)
        
        ## N.B. needed_displacements and upcoming_terrains are paired.
        ##      And their len is at most 2.
        upcoming_terrains = []
        for displacement in needed_displacements:
            #pos_terrain = pos_agent + action_id2ndarray[displacement]
            pos_terrain = self.pos + action_id2ndarray[displacement]
            value_terrain = self.view[pos_terrain[1], pos_terrain[0]]
            upcoming_terrains.append(value_terrain)

        #logging.debug(f"needed_displacements = {[constants02.action_id2str[mv] for mv in needed_displacements]}")
        logging.debug(f"needed_displacements = {[action_id2str[mv] for mv in needed_displacements]}")
        index = self.moins_severe_index(upcoming_terrains)
        logging.debug(f"upcoming_terrains = {upcoming_terrains}")
        #print(f"index = {index}")
        logging.debug(f"len(self.pos_impossible_golds) = {len(self.pos_impossible_golds)}")
        logging.debug(f"index is None: {index is None}")

        if not index is None:
            next_terrain = upcoming_terrains[index]
        else:
            logging.debug("BO TAY")
            return self.bo_2tay(needed_displacements)
        #print("next_terrain = {}, energy_agent = {}".format(next_terrain, energy_agent))
        if self.besoin_reposer(next_terrain):
            logging.debug("Take REST")
            return rest
        else:
            #print(f"(Final) needed_displacements = {needed_displacements}, index = {index}")
            #logging.debug(f"Take {constants02.action_id2str[needed_displacements[index]]}")
            logging.debug(f"Take {action_id2str[needed_displacements[index]]}")
            return needed_displacements[index]


    def find_pos_possible_golds(self, view):
        pos_golds = self.find_pos_golds(view)
        index_row_delete = []
        for pos_impossible in self.pos_impossible_golds:
            for i in range(pos_golds.shape[0]):
                pos = pos_golds[i]
                if np.array_equal(pos_impossible, pos):
                    index_row_delete.append(i)
                    break
        pos_possible_golds = np.delete(pos_golds, index_row_delete, 0)
        return pos_possible_golds

    def construct_pos_possible_golds(self):
        n_remain_mv = round(self.n_remain_steps * (2/3))
        pos_golds = self.trouver_pos_golds()
        index_row_delete = []
        for pos_impossible in self.pos_impossible_golds:
            for i in range(pos_golds.shape[0]):
                pos = pos_golds[i]
                if np.array_equal(pos_impossible, pos):
                    index_row_delete.append(i)
                    break
        pos_possible_golds = np.delete(pos_golds, index_row_delete, 0)
        return pos_possible_golds

    def update_pos_possible_golds(self, pos_rm_gold):
        # TODO
        pass

    def trouver_nearest_gold(self, pos_from=self.pos):
        distances = np.array([l1_dist(pos_gold, pos_from) for pos_gold in self.pos_possible_golds])
        #print(f"distances = {distances}")
        nearest_gold_index = np.argmin(distances)
        #print(f"nearest_gold_index = {nearest_gold_index}")
        return self.pos_possible_golds[nearest_gold_index]

    def find_nearest_gold(self, view, pos_agent):
        #pos_golds = self.find_pos_golds(view)
        pos_possible_golds = self.find_pos_possible_golds(view)
        distances = np.array([np.linalg.norm(pos-pos_agent, ord=1) for pos in pos_possible_golds])
        #print(f"distances = {distances}")
        nearest_gold_index = np.argmin(distances)
        #print(f"nearest_gold_index = {nearest_gold_index}")
        return pos_possible_golds[nearest_gold_index]

    def moins_severe_index(self, terrains):
        terrains = np.array(terrains)
        logging.debug(f"terrains.max() = {terrains.max()}")
        logging.debug(f"- swamp_energy[-1] = {- swamp_energy[-1]}")
        if terrains.max() <= - max_energy: # i.e. next terrain is fatal for sure 
            index = None
            return index
        if terrains.size == 1:
            index = 0
            return index
        else:
            if abs(terrains[0] - terrains[1]) < 0.2:
                ## This happens, for example, when I have in front of us swamp=-20.1 and forest=-20
                index = np.random.randint(2)
                return index
            else:
                index = np.argmax(terrains)
                return index

    def less_severe_index(self, terrains):
        terrains = np.array(terrains)
        terrains[terrains==-replace_swamp5] = -5
        terrains[terrains==-replace_swamp20] = -20
        logging.debug(f"terrains.max() = {terrains.max()}")
        logging.debug(f"- swamp_energy[-1] = {- swamp_energy[-1]}")
        if terrains.max() <= - swamp_energy[-1]: # i.e. next terrain is fatal swamp for sure
            index = None
            return index
        if terrains.size == 1:
            index = 0
            return index
        else:
            if terrains[0] == terrains[1]:
                index = np.random.randint(2)
                return index
            else:
                index = np.argmax(terrains)
                return index

    def besoin_reposer(self, next_terrain):
        if next_terrain > 0: # i.e. gold
            return self.energy <= dig_energy
        return self.energy + next_terrain <= 0

    def need_rest(self, next_terrain, energy):
        if next_terrain > 0: # i.e. gold
            return energy <= dig_energy
        return energy + next_terrain <= 0

    def bo_2tay(self, needed_displacements):
        ## Same as bo_tay() but newer version of it
        ## TODO: when shortest path is impossible
        legal_displacements = []
        #print(f"pos_agent = {pos_agent}")
        for mv in [up, down, left, right]:
            if mv in needed_displacements:
                continue
            else:
                pos_mv = self.pos + action_id2ndarray[mv]
                if out_of_map(pos_mv):
                    #print(f"{pos_mv} out of MAP")
                    continue
                elif self.view[pos_mv[1], pos_mv[0]] <= - max_energy:
                    logging.debug(f"{pos_mv} swamp ({self.view[pos_mv[1], pos_mv[0]]})")
                    continue
                else:
                    legal_displacements.append(mv)
        #print(f"legal_displacements = {legal_displacements}")
        #print(f"energy = {energy}")
        if len(legal_displacements) == 0:
            return rest
        chosen_mv = np.random.choice(legal_displacements)
        pos_next = self.pos + action_id2ndarray[chosen_mv]
        next_terrain = self.view[pos_next[1], pos_next[0]]
        if self.besoin_reposer(next_terrain):
            logging.debug("Take REST")
            return rest
        else:
            #logging.debug(f"Take {constants02.action_id2str[chosen_mv]}")
            logging.debug(f"Take {action_id2str[chosen_mv]}")
            return chosen_mv

    def bo_tay(self, needed_displacements, view, pos_agent, energy):
        ## TODO: when shortest path is impossible
        legal_displacements = []
        #print(f"pos_agent = {pos_agent}")
        for mv in [up, down, left, right]:
            if mv in needed_displacements:
                continue
            else:
                pos_mv = pos_agent + action_id2ndarray[mv]
                if out_of_map(pos_mv):
                    #print(f"{pos_mv} out of MAP")
                    continue
                elif view[pos_mv[1], pos_mv[0]] <= - swamp_energy[-1]:
                    logging.debug(f"{pos_mv} swamp ({view[pos_mv[1], pos_mv[0]]})")
                    continue
                else:
                    legal_displacements.append(mv)
        #print(f"legal_displacements = {legal_displacements}")
        #print(f"energy = {energy}")
        if len(legal_displacements) == 0:
            return rest
        chosen_mv = np.random.choice(legal_displacements)
        pos_next = pos_agent + action_id2ndarray[chosen_mv]
        next_terrain = view[pos_next[1], pos_next[0]]
        if self.need_rest(next_terrain, energy):
            logging.debug("Take REST")
            return rest
        else:
            #logging.debug(f"Take {constants02.action_id2str[chosen_mv]}")
            logging.debug(f"Take {action_id2str[chosen_mv]}")
            return chosen_mv

    #def policy_nearest_gold(view, energy, stepCount, pos_players):
    def policy_nearest_gold(self, view, energy, pos_players):
        pos_agent = pos_players[0]
        x, y = pos_agent
        standing_on_gold = view[y, x] > 0
        if standing_on_gold:
            self.pos_target_gold = None
            self.distance_to_target.clear()
            #if energy > constants02.dig_energy:
            if energy > dig_energy:
                #return constants02.dig
                return dig
            else:
                #return constants02.rest
                return rest
        ## TODO: Find best hyperparams here: 20, 7
        too_long = 30
        #too_long = 100
        long_ = 100
        if len(self.distance_to_target) > too_long:
            logging.warning("too_long happens")
            self.pos_impossible_golds.append(self.pos_target_gold)
        #elif len(self.distance_to_target) > long_ and self.distance_to_target[-1] >= min(list(self.distance_to_target)[:-1]):
        #    self.pos_impossible_golds.append(self.pos_target_gold)

        pos_nearest_gold = self.find_nearest_gold(view, pos_agent)
        logging.debug("")
        logging.debug(f"pos_agent = {pos_agent}")
        logging.debug(f"energy = {energy}")
        logging.debug(f"pos_nearest_gold = {pos_nearest_gold}")
        self.pos_target_gold = pos_nearest_gold
        self.distance_to_target.append(np.linalg.norm(self.pos_target_gold - pos_agent, ord=1))

        ## Try to follow the shortest path to gold
        needed_displacements = []
        vec_displacement = pos_nearest_gold - pos_agent
        # horizontal, i.e. x-movement
        if vec_displacement[0] > 0:
            #needed_displacements.extend(["right"]*vec_displacement[0])
            #needed_displacements.extend(["right"])
            needed_displacements.append(right)
        elif vec_displacement[0] < 0:
            #needed_displacements.extend(["left"]*vec_displacement[0])
            #needed_displacements.extend(["left"])
            needed_displacements.append(left)
        
        # vertical, i.e. y-movement
        if vec_displacement[1] > 0:
            #needed_displacements.extend(["down"]*vec_displacement[0])
            #needed_displacements.extend(["down"])
            needed_displacements.append(down)
        elif vec_displacement[1] < 0:
            #needed_displacements.extend(["up"]*vec_displacement[0])
            #needed_displacements.extend(["up"])
            needed_displacements.append(up)
        
        # N.B. needed_displacements and upcoming_terrains are paired.
        # And their len is at most 2.
        upcoming_terrains = []
        for displacement in needed_displacements:
            pos_terrain = pos_agent + action_id2ndarray[displacement]
            value_terrain = view[pos_terrain[1], pos_terrain[0]]
            upcoming_terrains.append(value_terrain)

        #logging.debug(f"needed_displacements = {[constants02.action_id2str[mv] for mv in needed_displacements]}")
        logging.debug(f"needed_displacements = {[action_id2str[mv] for mv in needed_displacements]}")
        index = self.less_severe_index(upcoming_terrains)
        logging.debug(f"upcoming_terrains = {upcoming_terrains}")
        #print(f"index = {index}")
        logging.debug(f"len(self.pos_impossible_golds) = {len(self.pos_impossible_golds)}")
        logging.debug(f"index is None: {index is None}")

        if not index is None:
            next_terrain = upcoming_terrains[index]
        else:
            logging.debug("BO TAY")
            return self.bo_tay(needed_displacements, view, pos_agent, energy)
        #print("next_terrain = {}, energy_agent = {}".format(next_terrain, energy_agent))
        if self.need_rest(next_terrain, energy):
            logging.debug("Take REST")
            return rest
        else:
            #print(f"(Final) needed_displacements = {needed_displacements}, index = {index}")
            #logging.debug(f"Take {constants02.action_id2str[needed_displacements[index]]}")
            logging.debug(f"Take {action_id2str[needed_displacements[index]]}")
            return needed_displacements[index]

    
    def find_gold_within_k_steps(self, pos_from=self.pos, k=3):
        # TODO
        pass

    def exist_gold_within_k_steps(self, pos_from=self.pos, k=3):
        exist = False
        for pos_gold in self.pos_possible_golds:
            if l1_dist(pos_gold, pos_from) <= k:
                exist = True
        return exist

    #def policy_00(self, view, energy, stepCount, pos_players):
    def policy_00(self, env):
        """
        """
        ## Update important attributes
        self.view, self.energy, self.stepCount, pos_players = env.get_non_RL_state_02()
        self.n_remain_steps = n_allowed_steps - self.stepCount
        self.pos = pos_players[0]
        self.pos_other_players = pos_players[1:]
        self.x, self.y = self.pos
        ## Update self.pos_possible_golds
        # TODO
        if self.stepCount == 0: # TODO: or == 1?
            self.construct_pos_possible_golds()
        else:
            self.update_pos_possible_golds()

        standing_on_gold = view[self.y, self.x] > 0
        if standing_on_gold:
            if energy > dig_energy:
                return dig
            else:
                return rest

        # TODO: Whether or not at Beginning, target most valuable gold

        # TODO: wehn len(self.pos_possible_golds) == 0
        if len(self.pos_possible_golds) == 0:
            ## Just go to nearest gold
            #self.pos_possible_golds.append()
            pass



        # TODO: Une fois the target gold epuise, il nous faut encore creuser de l'or qui sont autour
        if self.pos_target_gold is None:
            if self.exist_gold_within_k_steps():
                #return self.one_step_closer_to(self.trouver_nearest_gold())
                self.pos_target_gold = self.trouver_nearest_gold()
                return self.towards_target()
            else:
                self.pos_target_gold = self.trouver_most_valuable_gold()
                #return self.one_step_closer_to(self.pos_target_gold)
                return self.towards_target()
        else:
            #return self.one_step_closer_to(self.pos_target_gold)
            return self.towards_target()

