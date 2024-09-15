import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.estimate_depth import estimate_depth
from utils.bokeh import *
import hashlib

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
    bokeh_image = bokeh_effect_avg(image, depthmap=depth, focal_distance= focal_distance, kernel_size= kernel_size)
    bokeh_image = cv2.cvtColor(bokeh_image, cv2.COLOR_BGR2RGB)
    if resize_by:
        bokeh_image = cv2.resize(bokeh_image, (w, h))
    return bokeh_image


# Create a Gradio interface
iface = gr.Interface(
    fn=bokeh_effect,
    inputs=[
        gr.Image(type="numpy"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="focal_distance"),
        gr.Slider(minimum=3, maximum=15, step=2, label="Blur strength"),  
        gr.Slider(minimum=1, maximum=4, step=1, label="resize_by")
    ],
    outputs=gr.Image(type="numpy"),
    title="Depth of Field Effect",
    description="Upload an image and adjust parameters to see the depth of field effect."
)

# Launch the Gradio app
iface.launch()
