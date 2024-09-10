import cv2
import numpy as np
import torch



def split_into_patches(image, num_patches_height=10, num_patches_width=10):
    image = torch.from_numpy(image)
    H, W, C = image.shape
    patch_height = H//num_patches_height
    patch_width = W//num_patches_width
    patches = image.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)
    patches = patches.reshape(-1, C, patch_height, patch_width)
    return patches.numpy().transpose(0,2,3,1)
     
         
def gaussian_patch(kernel_size, sigma_patch):
    """Generates a Gaussian kernel."""
    h,w = sigma_patch.shape
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    term = np.exp(-(xx**2 + yy**2)/2).reshape(-1)
    term_tiled = np.tile(term, (h, w, 1))
    gaussian_patch = term_tiled * sigma_patch[:, :, np.newaxis]
    #kernel = np.exp(term/ (2. * sigma**2))
    return gaussian_patch

def patch_blurring(image_patch, gaussian_patch):
    result = np.sum(image_patch*gaussian_patch, axis=-1)
    result


def depth_of_field_effect(image, depthmap, focal_distance, blur_amount):
    weight_map = 1 - np.exp(-(depthmap - focal_distance)**2/ (2*blur_amount))
    # weight_map = np.abs(depthmap - focal_distance)
    kernel_size = int(2*blur_amount) + 1
    sigma = blur_amount
    if kernel_size%2 ==0:
        kernel_size +=1
    weight_map = weight_map / np.max(weight_map)

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size) , sigma)
    sharp_image = image.copy()
    weight_map = weight_map[..., np.newaxis]
    sharp_image = image * (1-weight_map) + blurred_image * weight_map
    return sharp_image

if __name__ == "__main__":
    kernel_size = 5
    sigma_patch = np.random.rand(100,100)
    gaussian_patch(kernel_size, sigma_patch)