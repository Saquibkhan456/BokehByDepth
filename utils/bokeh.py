import cv2
import numpy as np
import numexpr as ne
import time
def bokeh_effect_gaussian(image, depthmap, focal_distance, kernel_size = 5):
    weight_map = np.abs(depthmap - focal_distance)
    sigma_array = 3 * (0.001 + weight_map)**4
    gaussian_array = create_gaussian_array(kernel_size, sigma_array)
    image_array = prepare_image_array(image, kernel_size=kernel_size)
    s = time.time()
    prod = np.evaluate('image_array * avg_array[..., np.newaxis]')
    print(f"this took : {time.time() - s} seconds")
    blurred = (np.sum(prod, axis=0)).astype(np.uint8)
    return blurred
    

def bokeh_effect_avg(image, depthmap, focal_distance, kernel_size = 15):
    weight_map = np.abs(depthmap - focal_distance)
    kernel_matrix = (normalize_between_1_and_n(np.array(weight_map), kernel_size).astype(np.uint16))**2
    avg_array = create_avg_array2(kernel_matrix)
    print("blur kernel created")
    image_array = prepare_image_array2(image, kernel_size=kernel_size)
    print("image array created")
    blurred = np.einsum('ijkl,ijk->jkl', image_array, avg_array).astype(np.uint8)
    print("Done!")
    return blurred


def normalize_between_1_and_n(arr, n):
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = 1 + ((arr - arr_min) / (arr_max - arr_min)) * (n - 1)

    return normalized_arr

         
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
    # Create a range array of shape (10,) for comparison
    max_size = np.max(kernel_matrix)
    avg_matrix = np.zeros((kernel_matrix.shape[0], kernel_matrix.shape[1],max_size ), dtype=np.uint8)
    indices = np.arange(avg_matrix.shape[2])
    
    mask = indices < kernel_matrix[..., np.newaxis]
    
    avg_matrix[mask] = 1
    avg_matrix = avg_matrix/np.sum(avg_matrix, axis=2, keepdims=True)
    avg_matrix = np.roll(avg_matrix , (max_size//2 +1),axis=-1)
    return avg_matrix.transpose(2,0,1)


def create_avg_array2(kernel_matrix):
    # Get the maximum value in kernel_matrix
    max_size = np.max(kernel_matrix)
    
    # Create a 3D array of shape (rows, cols, max_size) initialized with zeros
    avg_matrix = np.zeros((kernel_matrix.shape[0], kernel_matrix.shape[1], max_size), dtype=np.uint8)
    
    # Create indices array for comparison
    indices = np.arange(max_size)
    
    # Vectorized mask to set values in avg_matrix
    mask = indices < kernel_matrix[..., np.newaxis]
    
    # Fill the avg_matrix with ones where the mask is True
    avg_matrix[mask] = 1
    
    # Sum along the third axis (max_size) and avoid division by zero
    sum_values = np.sum(avg_matrix, axis=2, keepdims=True)
    # Avoid division by zero
    sum_values[sum_values == 0] = 1
    # Normalize the avg_matrix
    avg_matrix = avg_matrix / sum_values
    
    # Roll the array along the last axis
    avg_matrix = np.roll(avg_matrix, (max_size // 2 + 1), axis=-1)
    
    # Transpose to shape (max_size, rows, cols)
    return avg_matrix.transpose(2, 0, 1)


def prepare_image_array(image, kernel_size):
    image_array = []
    padding = kernel_size//2
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    idx = np.arange(-kernel_size//2+1, kernel_size//2+1)[::-1]
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            part = np.roll(padded_image, (idx[j], idx[i]), axis=(1,0))
            image_array.append(part)
    image_array = np.array(image_array)
    result = image_array[:, padding:-padding, padding:-padding, :]
    return result

def prepare_image_array2(image, kernel_size):
    padding = kernel_size // 2
    # Pad the image using 'reflect' mode
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    # Use sliding_window_view to get a view of sliding windows of the padded image
    windows = np.lib.stride_tricks.sliding_window_view(padded_image, (kernel_size, kernel_size, image.shape[2]))
    
    # Reshape the windows to the desired output
    # New shape: (height, width, kernel_size, kernel_size, channels)
    result = windows[:, :, 0, ...]
    
    # Reshape result to match the original shape (kernel_size**2, height, width, channels)
    result = result.transpose(2, 3, 0, 1, 4).reshape(kernel_size**2, image.shape[0], image.shape[1], image.shape[2])

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