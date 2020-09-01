# This script contains all the get_state() methods of class MinerEnv
# that I have come to think of to help better train an agent.
#
# The higher the more up-to-date.

import constants

class MinerEnv:

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
        state = np.maximum(view, energyOnMap)
        return state


    def get_state(self):
        """
        BTC's idea about "local view", i.e. restrict the agent's vision range.
        This is aimed to simplify the training for the agent.
        """
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
            #self.pos_x_gold_first = self.state.mapInfo.golds[0]["posx"]
            #self.pos_y_gold_first = self.state.mapInfo.golds[0]["posy"]
            pos_closest_gold = non_RL_agent.find_closest_gold(self.get_traditional_state())
            self.pos_x_gold_first, self.pos_y_gold_first = pos_closest_gold

        logging.debug(f"self.pos_x_gold_first = {self.pos_x_gold_first}")
        DQNState.append(self.pos_x_gold_first - self.state.x)
        DQNState.append(self.pos_y_gold_first - self.state.y)
                
        DQNState = np.array(DQNState)

        return DQNState


    def get_state(self):
        """
        Modification from the original code, only `view` is changed, everything else untouched.
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

        DQNState = view.flatten().tolist()
        
        # Add position and energy of agent
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        # Add position of bots 
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])
                
        DQNState = np.array(DQNState)

        return DQNState
