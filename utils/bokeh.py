import numpy as np


def normalize_between_1_and_n(arr, n):
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = 1 + ((arr - arr_min) / (arr_max - arr_min)) * (n - 1)
    return normalized_arr

def bokeh_effect_avg(image, depthmap, focal_distance, kernel_size):
    weight_map = np.abs(depthmap - focal_distance)
    kernel_matrix = (normalize_between_1_and_n(np.array(weight_map), kernel_size).astype(np.uint16))**2
    avg_array = create_avg_array(kernel_matrix)
    print("blur kernel created")
    image_array = prepare_image_array(image, kernel_size=kernel_size)
    print("image array created")
    blurred = np.einsum('ijkl,ijk->jkl', image_array, avg_array).astype(np.uint8)
    print("Done!")
    return blurred


def create_gaussian_array(kernel_size, sigma_array):
    
    h,w = sigma_array.shape
    sigma_array = np.exp(1/(sigma_array**2))
    sigma_array[sigma_array == np.inf] = 100

    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    term = np.exp(-(xx**2 + yy**2)/2).reshape(-1)
    term_tiled = np.tile(term, (h, w, 1))
    gaussian_patch = term_tiled * sigma_array[:, :, np.newaxis]
    max_vals = np.sum(gaussian_patch, axis=2, keepdims=True)
    gaussian_patch = gaussian_patch/max_vals

    return gaussian_patch.transpose(2,0,1)

def create_avg_array(kernel_matrix):
    max_size = np.max(kernel_matrix)
    avg_matrix = np.zeros((kernel_matrix.shape[0], kernel_matrix.shape[1], max_size), dtype=np.uint8)
    indices = np.arange(max_size)
    mask = indices < kernel_matrix[..., np.newaxis]
    avg_matrix[mask] = 1
    sum_values = np.sum(avg_matrix, axis=2, keepdims=True)
    sum_values[sum_values == 0] = 1
    avg_matrix = avg_matrix / sum_values
    avg_matrix = np.roll(avg_matrix, (max_size // 2 + 1), axis=-1)
    return avg_matrix.transpose(2, 0, 1)

def prepare_image_array(image, kernel_size):
    padding = kernel_size // 2
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded_image, (kernel_size, kernel_size, image.shape[2]))
    result = windows[:, :, 0, ...].transpose(2, 3, 0, 1, 4).reshape(kernel_size**2, image.shape[0], image.shape[1], image.shape[2])
    return result