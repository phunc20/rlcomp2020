import numpy as np
import matplotlib.pyplot as plt
from constants import width, height, terrain_ids, n_px

# in RGB format
#gold = [99.3, 68.8, 1.0]
#silver = [47.8, 47.8, 47.8]
#blue = [8.2, 39.4, 95.6]
#green = [4.6, 43.8, 2.5]
#pink = [85.4, 47.7, 45.9]

#gold = np.array([99.3, 68.8, 1.0])
#silver = np.array([47.8, 47.8, 47.8])
#blue = np.array([8.2, 39.4, 95.6])
#green = np.array([4.6, 43.8, 2.5])
#pink = np.array([85.4, 47.7, 45.9])

silver = np.array([183, 179, 150])
blue = np.array([51, 147, 237])
green = np.array([17, 111, 17])
pink = np.array([253, 179, 176])
gold = np.array([242, 215, 35])
white = np.ones((3,), dtype=np.uint8)*255
black = np.zeros((3,), dtype=np.uint8)

def imshow(image, figsize=(7,7)):
    plt.figure(figsize=figsize)
    plt.imshow(image);
    #plt.grid(True);
    plt.yticks(range(image.shape[0]));
    plt.xticks(range(image.shape[1]));


def render_state_image(s):
    numerical_image = s[:n_px].reshape((height, width))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[numerical_image==-terrain_ids["forest"]] = green
    image[numerical_image==-terrain_ids["trap"]] = silver
    image[numerical_image==-terrain_ids["swamp"]] = blue
    image[numerical_image>0] = gold
    image[numerical_image==0] = pink
    return image

def pos_3x(xy):
    """
    args
        xy, ndarray
            shape = (2*k,)
    return
        xy_3x, ndarray
            shape = (2*k,)
    """
    xy_3x = xy*3 + 1
    return xy_3x

def prettier_render(s):
    primitive = render_state_image(s)
    primitive_3x = np.kron(primitive, np.ones((3,3,1), dtype=np.uint8))
    #print(f"primitive_3x.shape = {primitive_3x.shape}")

    # draw bots' position
    # The positions might be overwritten one over another
    n_bots = s[n_px+3:].size // 2
    bots_xy = s[n_px+3:].reshape((n_bots,2))
    bots_xy_3x = pos_3x(bots_xy)
    for bot_id, coord in enumerate(bots_xy_3x):
        print(f"bot {bot_id} at {bots_xy[bot_id]}")
        try:
            primitive_3x[coord[1], coord[0]] = black
        except IndexError as e:
            print(f"bot {bot_id} fell OUT OF MAP")
    
    # draw agent's position
    agent_xy = s[n_px:n_px+2]
    agent_xy_3x = pos_3x(agent_xy)
    #print(f"agent_xy_3x = {agent_xy_3x}")
    primitive_3x[agent_xy_3x[1], agent_xy_3x[0]] = white
    return primitive_3x


def gold_total(map_):
    #map_ = np.array(map_)
    return map_[map_ > 0].sum()

#def pictorial_state(obs):
#    pictorial = np.zeros((constants.height, constants.width, 2), dtype=np.float32)
#    # dtype=np.float32 because pictorial will later be carried into tensorflow CNN
#    pictorial[..., 0] = obs[:constants.n_px].reshape((constants.height, constants.width))
#    # position of agent: we put the energy value at the coordinate where stands the agent, the whole in the last channel.
#    pictorial[obs[constants.n_px], obs[constants.n_px+1], -1] = obs[constants.n_px+2]
#    # position of bots: we put -1 on the coord of the bots
#    for i in range(1, 3+1):
#        pictorial[obs[constants.n_px+(2*i+1)], obs[constants.n_px+(2*i+2)], -1] = -1
#    return pictorial


