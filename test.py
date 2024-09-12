import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.estimate_depth import estimate_depth
from utils.bokeh import *


def bokeh_effect(image, focal_distance, kernel_size, depth=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if depth is None:
        depth = np.array(estimate_depth(image, encoder='vitl'))
        
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = 1 - depth
        print("depth estimated")

    bokeh_image = bokeh_effect_avg(image, depthmap=depth, focal_distance= focal_distance, kernel_size= kernel_size)
    return bokeh_image, depth

if __name__ == "__main__":
    image = cv2.imread("assets/images/input.jpg")
    image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
    bokeh_image1,depth = bokeh_effect(image, focal_distance=0.01, kernel_size=9)
    bokeh_image2, _ = bokeh_effect(image, focal_distance=0.99, kernel_size=9, depth=depth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(141)
    plt.imshow(image)
    plt.axis('off')

    plt.title("Original", fontsize=8)
    plt.subplot(142)
    plt.imshow(depth)
    plt.axis('off')

    plt.title("estimated depth", fontsize=8)
    plt.subplot(143)
    plt.imshow(bokeh_image1)
    plt.axis('off')
    plt.title("near focus", fontsize=8)
    plt.subplot(144)
    plt.imshow(bokeh_image2)
    plt.axis('off')
    plt.title("far focus", fontsize=8)
    plt.savefig('output_image.jpg', format='jpg', bbox_inches='tight', pad_inches=0)
    plt.show()



