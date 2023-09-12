import random
from PIL import Image
import numpy as np


def draw_cell(index, classes, labels):
    # Create a palette containing 5 different colours representing 5 categories
    # palette = [(252, 175, 124), (135, 201, 195), (199, 199, 199), (4, 104, 107), (221, 160, 221)]
    palette = [(252, 175, 124), (135, 201, 195), (199, 199, 199), (4, 104, 107)]
    # for _ in range(np.max(classes)+1):
    #     # Generate random RGB colour values
    #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     palette.append(color)

    # Creating palette index images
    image = Image.new('RGB', (labels.shape[1], labels.shape[0]), 'black')

    for i, category in zip(index, classes):
        rows, cols = np.where(labels == i)
        coordinates = list(zip(rows, cols))
        for coord in coordinates:
            x = coord[1]
            y = coord[0]
            image.putpixel((x, y), palette[category])

    image.show()
    image.save('./cluster.png')

    return


