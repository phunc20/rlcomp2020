"""
methods' name starting with
01) find_: are older/deprecated methods
02) trouver_: are newer methods which usually have less input args


"""

import logging
#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)
import numpy as np
from scipy.ndimage import gaussian_filter
#import constants02
from constants02 import *
#from viz_utils02 import *
from collections import deque

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

    def valeur(self, pos_gold, n_approx_steps, k=3):
        #propre_valeur = 
        #extra_valeur = 
        # TODO: Take swamp into consideration
        logging.debug(f"(Inside valeur())")
        valeurs = []
        poids = []
        logging.debug(f"self.pos_possible_golds = {self.pos_possible_golds}")
        for pos_vang in self.pos_possible_golds:
            distance = l1_dist(pos_vang, pos_gold)
            if distance > k:
                continue
            else:
                poids.append(2**(-k))
                valeurs.append(self.view[pos_vang[1], pos_vang[0]])
        valeurs = np.array(valeurs)
        poids = np.array(poids)
        logging.debug(f"pos_gold = {pos_gold}")
        logging.debug(f"valeurs = {valeurs}")
        logging.debug(f"poids = {poids}")
        logging.debug(f"(valeurs*poids).sum() = {(valeurs*poids).sum()}")
        logging.debug(f"n_approx_steps*15 = {n_approx_steps*15}")
        total = max(0, (valeurs*poids).sum() - n_approx_steps*15)
        #total = max(0, (valeurs*poids).sum() / poids.sum() - n_approx_steps*15)
        if total == 0:
            logging.debug(f"{pos_gold} evaluated to 0")
        n_share = 1
        for pos_opponent in self.pos_other_players:
            if l1_dist(pos_opponent, pos_gold) <= k:
                n_share += 1
        total = total / n_share
        logging.debug(f"totally: {total}")
        return total

    #def evaluate_00(pos_gold, n_remain_steps):
    #def trouver_most_valuable_gold(pos_gold, n_remain_steps):
    #def trouver_most_valuable_gold(self, pos_from=self.pos, ratio=2/3):
    #def trouver_most_valuable_gold(self, pos_from=None, ratio=2/3):
    #    # TODO
    #    if pos_from is None:
    #        pos_from = self.pos
    #    logging.debug(f"pos_from = {pos_from}")
    #    n_remain_mv = round(self.n_remain_steps * ratio) # we assume every 3 steps need 1 rest, i.e. 2/3 of the time spending on real movement
    #    logging.debug(f"n_remain_mv = {n_remain_mv}")
    #    values = []
    #    duppelganger = self.pos_possible_golds.copy()
    #    logging.debug(f"duppelganger =\n{duppelganger}")
    #    self.pos_possible_golds = []
    #    try:
    #        for pos_gold in duppelganger:
    #            distance = l1_dist(pos_gold, pos_from)
    #            n_approx_needed_steps = round(distance / ratio)
    #            #if distance >= n_remain_mv:
    #            if n_approx_needed_steps >= self.n_remain_steps:
    #                logging.debug(f"pos_gold {pos_gold} distance {int(distance):2d} too far")
    #                self.pos_impossible_golds.append(pos_gold)
    #                #self.update_pos_possible_golds(pos_gold)
    #                #self.pos_possible_golds.append(pos_gold)
    #            else:
    #                logging.debug(f"pos_gold {pos_gold} distance {int(distance):2d} reachable")
    #                ## TODO: n_left_steps to be tuned
    #                n_left_steps = max(0, self.n_remain_steps - n_approx_needed_steps - 1)
    #                if n_left_steps == 0:
    #                    logging.debug(f"n_left_steps = {n_left_steps}")
    #                value = self.valeur(pos_gold, n_left_steps)
    #                values.append(value)
    #                self.pos_possible_golds.append(pos_gold)
    #        values = np.array(values)
    #        self.pos_possible_golds = np.array(self.pos_possible_golds, dtype=np.int8)
    #        #index = np.argmax(values, axis=)
    #        index = np.argmax(values)
    #        return self.pos_possible_golds[index]
    #    except ValueError as e:
    #        logging.debug(f"{e}")
    #        return None

    def trouver_most_valuable_gold(self, pos_from=None, ratio=2/3):
        if pos_from is None:
            pos_from = self.pos
        self.mis_a_jour()
        logging.debug(f"pos_from = {pos_from}")
        n_remain_mv = round(self.n_remain_steps * ratio) # we assume every 3 steps need 1 rest, i.e. 2/3 of the time spending on real movement
        logging.debug(f"n_remain_mv = {n_remain_mv}")
        values = []
        #duppelganger = self.pos_possible_golds.copy()
        #logging.debug(f"duppelganger =\n{duppelganger}")
        #self.pos_possible_golds = []
        try:
            for pos_gold in self.pos_possible_golds:
                #distance = l1_dist(pos_gold, pos_from)
                #n_approx_needed_steps = round(distance / ratio)
                ##if distance >= n_remain_mv:
                #if n_approx_needed_steps < self.n_remain_steps:
                #    logging.debug(f"pos_gold {pos_gold} distance {int(distance):2d} reachable")
                #    ## TODO: n_left_steps to be tuned
                #    n_left_steps = max(0, self.n_remain_steps - n_approx_needed_steps - 1)
                #    if n_left_steps == 0:
                #        logging.debug(f"n_left_steps = {n_left_steps}")
                #    value = self.valeur(pos_gold, n_left_steps)
                #    values.append(value)
                #    self.pos_possible_golds.append(pos_gold)
                distance = l1_dist(pos_gold, pos_from)
                n_steps_left = self.n_remain_steps - distance
                value = self.valeur(pos_gold, n_steps_left)
                values.append(value)
            values = np.array(values)
            #self.pos_possible_golds = np.array(self.pos_possible_golds, dtype=np.int8)
            #index = np.argmax(values, axis=)
            index = np.argmax(values)
            return self.pos_possible_golds[index]
        except ValueError as e:
            logging.debug(f"{e}")
            return None

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
        logging.debug(f"(Entering towards_target())")
        logging.debug(f"self.pos_target_gold = {self.pos_target_gold}, self.pos = {self.pos}")
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
        logging.debug("(Inside construct_pos_possible_golds())")
        n_remain_mv = round(self.n_remain_steps * (2/3))
        logging.debug(f"n_remain_mv = {n_remain_mv}")
        pos_golds = self.trouver_pos_golds()
        self.pos_possible_golds = []
        for pos_gold in pos_golds:
            if l1_dist(pos_gold, self.pos) < n_remain_mv:
                self.pos_possible_golds.append(pos_gold)
        self.pos_possible_golds = np.array(self.pos_possible_golds, dtype=np.int8)
        logging.debug("(after construction)")
        logging.debug(f"self.pos_possible_golds = {self.pos_possible_golds}")
        #index_row_delete = []
        #for pos_impossible in self.pos_impossible_golds:
        #    for i in range(pos_golds.shape[0]):
        #        pos = pos_golds[i]
        #        if np.array_equal(pos_impossible, pos):
        #            index_row_delete.append(i)
        #            break
        #self.pos_possible_golds = np.delete(pos_golds, index_row_delete, 0)
        #return self.pos_possible_golds

    def update_pos_possible_golds(self, pos_rm_gold):
        """
        Here I choose to use np.delete() to do the job.

        Another choice would be to make a copy of self.pos_possible_golds,
        and then empty self.pos_possible_golds, looping thru its copy,
        and reconstruct self.pos_possible_golds so that pos_rm_gold
        is excluded.
        """
        index = []
        ## Note that if
        ## K = np.array([
        ##        [0, 1],
        ##        [2, 3],
        ##        [4, 5],
        ##        [6, 7]])
        ## then
        ## In [55]: np.delete(K, -1, axis=0)
        ## Out[55]:
        ## array([[0, 1],
        ##        [2, 3],
        ##        [4, 5]])
        ## 
        ## In [56]: np.delete(K, 1, axis=0)
        ## Out[56]:
        ## array([[0, 1],
        ##        [4, 5],
        ##        [6, 7]])
        ## 
        ## In [62]: np.delete(K, [], axis=0)
        ## Out[62]:
        ## array([[0, 1],
        ##        [2, 3],
        ##        [4, 5],
        ##        [6, 7]])
        for i, pos in enumerate(self.pos_possible_golds):
            if np.array_equal(pos, pos_rm_gold):
                index = i
                break
        self.pos_possible_golds = np.delete(self.pos_possible_golds, index, axis=0)

    def mis_a_jour(self):
        # Cf. update_pos_possible_golds()
        indices = []
        for i, pos in enumerate(self.pos_possible_golds):
            terrain = self.view[pos[1], pos[0]]
            distance = l1_dist(pos, self.pos)
            if terrain <= 0:
                indices.append(i)
            elif distance >= self.n_remain_steps:
                indices.append(i)
        self.pos_possible_golds = np.delete(self.pos_possible_golds, indices, axis=0)

    def trouver_nearest_gold(self, pos_from=None):
        if pos_from is None:
            pos_from=self.pos
        self.mis_a_jour()
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

    
    def find_gold_within_k_steps(self, pos_from=None, k=3):
        if pos_from is None:
            pos_from=self.pos
        # TODO
        pass

    def exist_gold_within_k_steps(self, pos_from=None, k=3):
        if pos_from is None:
            pos_from=self.pos
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
        logging.debug(f"self.stepCount = {self.stepCount}")
        if self.stepCount == 0: # TODO: or == 1?
            self.construct_pos_possible_golds()
        else:
            #self.update_pos_possible_golds()
            pass

        terrain_here = self.view[self.y, self.x]
        standing_on_gold = terrain_here > 0
        if standing_on_gold:
            if self.energy > dig_energy:
                # No Good
                #if terrain_here <= 50: # i.e. this will be the last dig
                #    self.pos_target_gold = None
                return dig
            else:
                return rest

        ## We need to empty self.pos_target_gold once it contains no gold
        if (not self.pos_target_gold is None) and (self.view[self.pos_target_gold[1], self.pos_target_gold[0]] <= 0):
            logging.debug(f"Epuise self.pos_target_gold = {self.pos_target_gold}, amount = {self.view[self.pos_target_gold[1], self.pos_target_gold[0]]}")
            self.update_pos_possible_golds(self.pos_target_gold)
            self.pos_target_gold = None

        # TODO: Whether or not at Beginning, target most valuable gold

        # TODO: wehn len(self.pos_possible_golds) == 0
        if len(self.pos_possible_golds) == 0:
            ## Just go to nearest gold
            #self.pos_possible_golds.append()
            logging.debug(f"len(self.pos_possible_golds) equals 0")
            pass



        # TODO: Une fois the target gold epuise, il nous faut encore creuser de l'or qui sont autour
        logging.debug(f"self.pos_target_gold = {self.pos_target_gold}")
        if self.pos_target_gold is None:
            if self.exist_gold_within_k_steps(k=1):
                #return self.one_step_closer_to(self.trouver_nearest_gold())
                logging.debug(f"exist_gold_within_k_steps")
                self.pos_target_gold = self.trouver_nearest_gold()
                logging.debug(f"trouver_nearest_gold = {self.pos_target_gold}")
                if self.pos_target_gold is None:
                    return rest
                else:
                    return self.towards_target()
            else:
                logging.debug(f"NOT exist_gold_within_k_steps")
                self.pos_target_gold = self.trouver_most_valuable_gold()
                #return self.one_step_closer_to(self.pos_target_gold)
                #return self.towards_target()
                if self.pos_target_gold is None:
                    return rest
                else:
                    return self.towards_target()
        else:
            #return self.one_step_closer_to(self.pos_target_gold)
            return self.towards_target()

