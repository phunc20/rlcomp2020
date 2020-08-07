import numpy as np
import matplotlib.pyplot as plt
from constants import width, height, terrain_ids

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


def imshow(image):
    plt.figure(figsize=(7,7))
    plt.imshow(image);
    #plt.grid(True);
    plt.yticks(range(image.shape[0]));
    plt.xticks(range(image.shape[1]));


def render_state_image(s):
    numerical_image = s[:width*height].reshape((height, width))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[numerical_image==-terrain_ids["forest"]] = green
    image[numerical_image==-terrain_ids["trap"]] = silver
    image[numerical_image==-terrain_ids["swamp"]] = blue
    image[numerical_image>0] = gold
    image[numerical_image==0] = pink
    imshow(image)



