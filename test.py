import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.estimate_depth import estimate_depth
from utils.bokeh import *
import hashlib
import time

cache = {}
def hash_image(image):
    # Convert the image to bytes and hash it
    _, buffer = cv2.imencode('.png', image)
    return hashlib.sha256(buffer).hexdigest()

def bokeh_effect(image, focal_distance, kernel_size,  resize_by = 1, depth=None,encoder = 'vits'):
    
    h,w,c = image.shape
    image_hash = hash_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (image.shape[1]//resize_by, image.shape[0]//resize_by))
    if depth is None or image_hash not in cache :
        depth = np.array(estimate_depth(image, encoder=encoder))
        
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = 1 - depth
        print("depth estimated")
        depth = (depth)**2
        cache[image_hash] = depth
        print("depth saved as cache")
    elif depth is not None and image_hash in cache :
        depth = cache[image_hash]
        print("using cached depth")
    s = time.time()
    bokeh_image = bokeh_effect_avg(image, depthmap=depth, focal_distance= focal_distance, kernel_size= kernel_size)
    print(f'this took : {time.time() - s} seconds')
    bokeh_image = cv2.cvtColor(bokeh_image, cv2.COLOR_BGR2RGB)
    if resize_by:
        bokeh_image = cv2.resize(bokeh_image, (w, h))
    return bokeh_image

if __name__ == "__main__":
    image = cv2.imread("assets/images/bench.jpg")
    image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
    bokeh_image1,depth = bokeh_effect(image, focal_distance=0.1, kernel_size=11)
    bokeh_image2, _ = bokeh_effect(image, focal_distance=0.99, kernel_size=11, depth=depth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(141)
    plt.imshow(image)
    plt.axis('off')

    plt.title("Original", fontsize=8)
    plt.subplot(142)
    plt.imshow(depth, cmap="jet")
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
    plt.savefig('teaser2.jpg', format='jpg', bbox_inches='tight', dpi=1000, pad_inches=0)
    plt.show()



