import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.estimate_depth import estimate_depth
from src.utils import *


def bokeh_effect(image, focal_distance, kernel_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.array(estimate_depth(image, encoder='vitl'))
    
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = 1 - depth
    print("depth estimated")

    bokeh_image = bokeh_effect_avg(image, depthmap=depth, focal_distance= focal_distance, kernel_size= kernel_size)
    return bokeh_image

if __name__ == "__main__":
    image = cv2.imread("assets/images/input.jpg")
    image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
    bokeh_image1 = bokeh_effect(image, focal_distance=0.1, kernel_size=7)
    bokeh_image2 = bokeh_effect(image, focal_distance=0.8, kernel_size=7)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(bokeh_image1)
    plt.title("near focus")
    plt.subplot(122)
    plt.imshow(bokeh_image2)
    plt.title("far focus")
    plt.show()



