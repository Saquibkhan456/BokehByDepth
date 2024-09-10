import cv2
import numpy as np
import torch

def bokeh_effect(image, depthmap, focal_distance, kernel_size = 5):
    weight_map = 1 - np.exp(-(depthmap - focal_distance)**2/ (2*kernel_size))
    weight_map = weight_map / np.max(weight_map)
    sigma_array = 2 * weight_map**2
    gaussian_array = create_gaussian_array(kernel_size, sigma_array)
    image_array = prepare_image_array(image)
    blurred = np.sum(image_array * gaussian_array[..., np.newaxis], axis=-1).astype(np.uint8)
    return blurred
    

# def split_into_patches(image_array, num_patches_height=10, num_patches_width=10):
#     image_array = torch.from_numpy(image_array)
#     K,H, W, C = image_array.shape
#     patch_height = H//num_patches_height
#     patch_width = W//num_patches_width
#     patches = image_array.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
#     patches = patches.reshape(K,num_patches_height * num_patches_width, C, patch_height, patch_width)
#     return patches.numpy().transpose(0,1,3,2,1)
     
         
def create_gaussian_array(kernel_size, sigma_array):
    """Generates a Gaussian kernel."""
    h,w = sigma_array.shape
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    term = np.exp(-(xx**2 + yy**2)/2).reshape(-1)
    term_tiled = np.tile(term, (h, w, 1))
    gaussian_patch = term_tiled * sigma_array[:, :, np.newaxis]
    return gaussian_patch

# def stitch_patches_together(image_patches):
#     final_result = np.zeros((image_patches.shape[0] * image_patches.shape[2],
#                              image_patches.shape[1] * image_patches.shape[3]), dtype=np.uint8)
#     for i in range(image_patches.shape[0]):
#         for j in range(image_patches.shape[0]):
#             final_result[i:image_patches.shape[2], j:image_patches.shape[3]] = image_patches[i,j]
#     return final_result



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
    create_gaussian_array(kernel_size, sigma_patch)