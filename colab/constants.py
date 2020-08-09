import numpy as np

width = 21
height = 9

terrain_ids = {
    "land": 0,
    "forest": 1,
    "trap": 2,
    "swamp": 3,
}

terrain_names = {value: key for key, value in terrain_ids.items()}

available_actions = {
    "up": '2',
    "down": '3',
    "left": '0',
    "right": '1',
    "rest": '4',
    "dig": '5',
}

punishments = {
    "land": 1,
    "gold": 4,
    "trap": 10,
    "forest": 20,
    "swamp": 40,
}


