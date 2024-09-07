import numpy as np
import matplotlib.pyplot as plt
import cv2
from .estimate_depth import estimate_depth
from .utils import depth_of_field_effect


def bokeh_effect(image, focal_distance, blur_strength):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.array(estimate_depth(image, encoder='vitl'))
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    print("depth estimated")

    bokeh_image = depth_of_field_effect(image, depthmap=depth, focal_distance= focal_distance, blur_amount = blur_strength)
    return bokeh_image

if __name__ == "__main__":
    image = cv2.imread("images/ring.jpg")
    bokeh_image = bokeh_effect(image, focal_distance=0.5, blur_strength=7)
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Input")
    plt.subplot(122)
    plt.imshow(bokeh_image)
    plt.title("Bokeh output")
    plt.show()



