import cv2
import numpy as np
import torch



def split_into_patches(image_array, num_patches_height=10, num_patches_width=10):
    image_array = torch.from_numpy(image_array)
    K,H, W, C = image_array.shape
    patch_height = H//num_patches_height
    patch_width = W//num_patches_width
    patches = image_array.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    patches = patches.reshape(K,num_patches_height * num_patches_width, C, patch_height, patch_width)
    return patches.numpy().transpose(0,1,3,2,1)
     
         
def gaussian_patch(kernel_size, sigma_patch):
    """Generates a Gaussian kernel."""
    h,w = sigma_patch.shape
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    term = np.exp(-(xx**2 + yy**2)/2).reshape(-1)
    term_tiled = np.tile(term, (h, w, 1))
    gaussian_patch = term_tiled * sigma_patch[:, :, np.newaxis]
    return gaussian_patch

def patch_blurring(image_patch_array, gaussian_patch):
    result = np.sum(image_patch_array*gaussian_patch, axis=-1)
    result


def prepare_image_array(image, kernel_size):
    image_array = []
    padding = kernel_size//2
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    idx = np.arange(-kernel_size//2+1, kernel_size//2+1)
    for i in range(kernel_size):
        for j in range(kernel_size):
            part = np.roll(padded_image, (idx[j], idx[i]), axis=(1,0))
            image_array.append(part)
    result = image_array[:, padding:-padding, padding:-padding, :]
    return result


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