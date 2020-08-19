import numpy as np
from constants import *
from viz_utils import *
import logging
#logging.basicConfig(filename='example.log',level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)

def find_largest_gold(s):
    numerical_image = s[:n_px].reshape((height, width))
    #print(f"\nnumerical_image =\n{numerical_image}")
    #row_col_largest = np.argmax(numerical_image)
    row_col_largest = np.unravel_index(np.argmax(numerical_image, axis=None), numerical_image.shape)
    #print(f"row_col_largest =\n{row_col_largest}")
    pos_largest = np.array(row_col_largest[::-1])
    return pos_largest

def find_pos_golds(s):
    numerical_image = s[:n_px].reshape((height, width))
    #print(f"\nnumerical_image =\n{numerical_image}")
    row_col_golds = np.argwhere(numerical_image>0)
    pos_golds = np.zeros_like(row_col_golds)
    pos_golds[:,0], pos_golds[:,1] = row_col_golds[:,1], row_col_golds[:,0]
    return pos_golds


def find_closest_gold(s):
    #numerical_image = s[:n_px].reshape((height, width))
    ##print(f"\nnumerical_image =\n{numerical_image}")
    #row_col_golds = np.argwhere(numerical_image>0)
    #pos_golds = np.zeros_like(row_col_golds)
    #pos_golds[:,0], pos_golds[:,1] = row_col_golds[:,1], row_col_golds[:,0]
    pos_golds = find_pos_golds(s)
    #print(f"row_col_golds =\n{row_col_golds}")
    #print(f"pos_golds =\n{pos_golds}")
    pos_agent = s[n_px:n_px+2]
    #print(f"pos_agent = {pos_agent}")
    distances = np.array([np.linalg.norm(pos-pos_agent, ord=1) for pos in pos_golds])
    #print(f"distances = {distances}")
    closest_gold_index = np.argmin(distances)
    #print(f"closest_gold_index = {closest_gold_index}")
    return pos_golds[closest_gold_index]

def find_worthiest_gold(s):
    numerical_image = s[:n_px].reshape((height, width))
    pos_agent = s[n_px:n_px+2]
    farest_possible = width + height - 2
    big_gold = 1000
    coeff = big_gold*farest_possible/(farest_possible-1)
    pos_golds = find_pos_golds(s)
    worth_golds = np.array([
        numerical_image[pos[1], pos[0]] + coeff/max(1, np.linalg.norm(pos-pos_agent, ord=1)) for pos in pos_golds
    ])
    index_worthiest = np.argmax(worth_golds)
    return pos_golds[index_worthiest]

def need_rest(next_terrain, energy):
    if next_terrain == "gold":
        if energy <= punishments[next_terrain]:
            return True
        else:
            return False
    elif next_terrain == "land":
        if energy <= punishments[next_terrain]:
            return True
        else:
            return False
    elif next_terrain == "trap":
        if energy <= punishments[next_terrain]:
            return True
        else:
            return False
    elif next_terrain == "forest":
        if energy <= punishments[next_terrain]:
            return True
        else:
            return False
    elif next_terrain == "swamp":
        if energy <= punishments[next_terrain]:
            return True
        else:
            return False

def less_severe_index(terrains):
    """
    args
        terrains, list
            a list of two terrains
    return
        terrain, str
        index, int
            0 or 1
    """
    #if terrains[0] == terrains[1]:
    #    return terrains[0]
    # Just return str's of terrain in increasing severity order
    if "land" in terrains:
        if terrains[0] == "land":
            index = 0
        else:
            index = 1
        return index
    if "trap" in terrains:
        if terrains[0] == "trap":
            index = 0
        else:
            index = 1
        return index
    if "forest" in terrains:
        if terrains[0] == "forest":
            index = 0
        else:
            index = 1
        return index
    return 0

def inverse_displacement(displacement):
    if displacement == "up":
        return "down"
    if displacement == "down":
        return "up"
    if displacement == "left":
        return "right"
    if displacement == "right":
        return "left"

def greedy_policy(s, how_gold=find_closest_gold, prev_displacement=None):
    #imshow(prettier_render(s))
    numerical_image = s[:n_px].reshape((height, width))
    #pos_closest_gold = find_closest_gold(s)
    pos_closest_gold = how_gold(s)
    #print(f"pos_closest_gold = {pos_closest_gold}")
    pos_agent = s[n_px:n_px+2]
    energy_agent = s[n_px+2]
    #print(f"pos_agent = {pos_agent}")
    #print(f"energy_agent = {energy_agent}")
    # If the agent stands on some gold
    if np.array_equal(pos_agent, pos_closest_gold):
        if energy_agent > 5:
            #print("digging...")
            return available_actions["dig"]
        else:
            return available_actions["rest"]
    
    # Walk to the closest gold
    needed_displacements = []
    vec_displacement = pos_closest_gold - pos_agent
    # horizontal, i.e. x-movement
    if vec_displacement[0] > 0:
        #needed_displacements.extend(["right"]*vec_displacement[0])
        needed_displacements.extend(["right"])
    elif vec_displacement[0] < 0:
        #needed_displacements.extend(["left"]*vec_displacement[0])
        needed_displacements.extend(["left"])
    
    # vertical, i.e. y-movement
    if vec_displacement[1] > 0:
        #needed_displacements.extend(["down"]*vec_displacement[0])
        needed_displacements.extend(["down"])
    elif vec_displacement[1] < 0:
        #needed_displacements.extend(["up"]*vec_displacement[0])
        needed_displacements.extend(["up"])
    
    # N.B. needed_displacements and upcoming_terrains are paired.
    # And their len is at most 2.
    upcoming_terrains = []
    for displacement in needed_displacements:
        if displacement == "right":
            pos_terrain = pos_agent + np.array([1,0])
        elif displacement == "left":
            pos_terrain = pos_agent + np.array([-1,0])
        elif displacement == "down":
            pos_terrain = pos_agent + np.array([0,1])
        elif displacement == "up":
            pos_terrain = pos_agent + np.array([0,-1])
        logging.debug(f"pos_terrain = {pos_terrain}")
        print(f"pos_terrain = {pos_terrain}")
        id_terrain = -numerical_image[pos_terrain[1], pos_terrain[0]]
        logging.debug(f"id_terrain = {id_terrain}")
        print(f"id_terrain = {id_terrain}")
        name_terrain = terrain_names[id_terrain] if id_terrain >= 0 else "gold"
        upcoming_terrains.append(name_terrain)

    if len(upcoming_terrains) == 2:
        index = less_severe_index(upcoming_terrains)
    else:
        index = 0
    next_terrain = upcoming_terrains[index]
    ##################################
    ## BEGIN: Never step into swamp ##
    ##################################
    if next_terrain == "swamp":
        permissible_displacements = {"up", "down", "left", "right"}
        # Avoid infinite loop, e.g. [up, down, up, down, ...]
        if prev_displacement:
            permissible_displacements.remove(inverse_displacement(prev_displacement))

        if pos_agent[0] == 0: # x = 0
            permissible_displacements.remove("left")
        elif pos_agent[0] == width-1:
            permissible_displacements.remove("right")

        if pos_agent[1] == 0: # y = 0
            permissible_displacements.remove("up")
        elif pos_agent[1] == height-1:
            permissible_displacements.remove("down")
        permissible_displacements = permissible_displacements - set(needed_displacements)
        logging.debug(f"permissible_displacements = {permissible_displacements}")
        print(f"permissible_displacements = {permissible_displacements}")
        if len(permissible_displacements) == 0:
            pass
        else:
            # This will then be a singleton list
            chosen = np.random.choice(list(permissible_displacements))
            logging.debug(f"chosen = {chosen}")
            print(f"chosen = {chosen}")
            needed_displacements = [chosen]
            index = 0
            if chosen == "right":
                pos_terrain = pos_agent + np.array([1,0])
            elif chosen == "left":
                pos_terrain = pos_agent + np.array([-1,0])
            elif chosen == "down":
                pos_terrain = pos_agent + np.array([0,1])
            elif chosen == "up":
                pos_terrain = pos_agent + np.array([0,-1])
            id_terrain = -numerical_image[pos_terrain[1], pos_terrain[0]]
            next_terrain = terrain_names[id_terrain] if id_terrain >= 0 else "gold"
    ################################
    ## END: Never step into swamp ##
    ################################
    logging.debug("next_terrain = {}, energy_agent = {}".format(next_terrain, energy_agent))
    print("next_terrain = {}, energy_agent = {}".format(next_terrain, energy_agent))
    if need_rest(next_terrain, energy_agent):
        return available_actions["rest"]
    else:
        logging.debug(f"(Final) needed_displacements = {needed_displacements}, index = {index}, pos_agent = {pos_agent}")
        print(f"(Final) needed_displacements = {needed_displacements}, index = {index}, pos_agent = {pos_agent}")
        return available_actions[needed_displacements[index]]

