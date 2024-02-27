'''
AUTHORS:
ERTURK OCAK 150200048
BATURALP INCE 150200738
'''

import numpy as np
import cv2
import math


img_as_BGR = cv2.imread("q3_noisy_img_sigma_5.png", cv2.IMREAD_COLOR) # read the image created in the previous question
noisy_image_sigma_5 = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.float64) 

img_as_BGR = cv2.imread("q3_noisy_img_sigma_10.png", cv2.IMREAD_COLOR) # read the image created in the previous question 
noisy_image_sigma_10 = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.float64) 

img_as_BGR = cv2.imread("q3_noisy_img_sigma_20.png", cv2.IMREAD_COLOR) # read the image created in the previous question
noisy_image_sigma_20 = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.float64) 



def change_to_grayscale(img_as_RGB): # this function is explained in previous questions
    
    height = img_as_RGB.shape[0] 
    width = img_as_RGB.shape[1] 
    grayscaled_img = np.zeros((height,width), dtype= np.float64) 
    for i in range(height):
        for j in range(width):
            red = img_as_RGB[i][j][0] 
            green = img_as_RGB[i][j][1]  
            blue = img_as_RGB[i][j][2]
            grayscale_value = round(0.299 * red + 0.587 * green + 0.114 * blue) 
            grayscaled_img[i][j] = grayscale_value 
    return grayscaled_img

noisy_image_sigma_5 = change_to_grayscale(noisy_image_sigma_5) # grayscale all the input images
noisy_image_sigma_10 = change_to_grayscale(noisy_image_sigma_10)
noisy_image_sigma_20 = change_to_grayscale(noisy_image_sigma_20)

def zero_padding_grayscaled(grayscaled_img, padding):   # this function is explained in previous questions
    stride = int((padding-1)/2) 
    number_of_rows = len(grayscaled_img)  
    number_of_columns = len(grayscaled_img[0])

    holder_matrix = [  
        [0 for col in range(number_of_columns + stride * 2)]
        for row in range(number_of_rows + stride * 2)
    ]

    for row in range(number_of_rows):
        for col in range(number_of_columns):
            holder_matrix[row + stride][col + stride] = grayscaled_img[row][col] 

    return np.array(holder_matrix)

def x5_gaussian_filtering(grayscaled_img, stride): # this function is explained in previous questions
    height, width = grayscaled_img.shape
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride)
    gaussian_filtered_img = np.array((zero_padded_grayscaled_img), dtype= np.float64)


    mask = np.array([[1, 4, 6, 4, 1],
                     [4, 16, 24, 16, 4],
                     [6, 24, 36, 24, 6],
                     [4, 16, 24, 16, 4],
                     [1, 4, 6, 4, 1]])
    mask = mask / 256 

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = (
                        zero_padded_grayscaled_img[i + k - 2, j + k - 2] * mask[0, 0]
                        + zero_padded_grayscaled_img[i + k - 2, j + k - 1] * mask[0, 1]
                        + zero_padded_grayscaled_img[i + k - 2, j + k] * mask[0, 2]
                        + zero_padded_grayscaled_img[i + k - 2, j + k + 1] * mask[0, 3]
                        + zero_padded_grayscaled_img[i + k - 2, j + k + 2] * mask[0, 4]
                        + zero_padded_grayscaled_img[i + k - 1, j + k - 2] * mask[1, 0]
                        + zero_padded_grayscaled_img[i + k - 1, j + k - 1] * mask[1, 1]
                        + zero_padded_grayscaled_img[i + k - 1, j + k] * mask[1, 2]
                        + zero_padded_grayscaled_img[i + k - 1, j + k + 1] * mask[1, 3]
                        + zero_padded_grayscaled_img[i + k - 1, j + k + 2] * mask[1, 4]
                        + zero_padded_grayscaled_img[i + k, j + k - 2] * mask[2, 0]
                        + zero_padded_grayscaled_img[i + k, j + k - 1] * mask[2, 1]
                        + zero_padded_grayscaled_img[i + k, j + k] * mask[2, 2]
                        + zero_padded_grayscaled_img[i + k, j + k + 1] * mask[2, 3]
                        + zero_padded_grayscaled_img[i + k, j + k + 2] * mask[2, 4]
                        + zero_padded_grayscaled_img[i + k + 1, j + k - 2] * mask[3, 0]
                        + zero_padded_grayscaled_img[i + k + 1, j + k - 1] * mask[3, 1]
                        + zero_padded_grayscaled_img[i + k + 1, j + k] * mask[3, 2]
                        + zero_padded_grayscaled_img[i + k + 1, j + k + 1] * mask[3, 3]
                        + zero_padded_grayscaled_img[i + k + 1, j + k + 2] * mask[3, 4]
                        + zero_padded_grayscaled_img[i + k + 2, j + k - 2] * mask[4, 0]
                        + zero_padded_grayscaled_img[i + k + 2, j + k - 1] * mask[4, 1]
                        + zero_padded_grayscaled_img[i + k + 2, j + k] * mask[4, 2]
                        + zero_padded_grayscaled_img[i + k + 2, j + k + 1] * mask[4, 3]
                        + zero_padded_grayscaled_img[i + k + 2, j + k + 2] * mask[4, 4]
                )
            gaussian_filtered_img[i+k][j+k] = round(weighted_mean_around)
    return gaussian_filtered_img

def x_derivative_of_img_using_sobel(grayscaled_img,stride): # this function is explained in previous questions
    height, width = grayscaled_img.shape
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride)
    x_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64)
    mask = np.array([[1,2,1],
                     [0,0,0],
                     [-1,-2,-1]])

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            x_derivative_of_img[i+k][j+k] = weighted_mean_around
    
    return x_derivative_of_img

def y_derivative_of_img_using_sobel(grayscaled_img,stride): # this function is explained in previous questions
    height, width = grayscaled_img.shape
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride)
    y_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64)
    mask = np.array([[1,0,-1],
                     [2,0,-2],
                     [1,0,-1]])

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            y_derivative_of_img[i+k][j+k] = weighted_mean_around
    
    return y_derivative_of_img

def gradient_magnitude(x_derivative_of_img, y_derivative_of_img): # this function is explained in previous questions
    height,width = x_derivative_of_img.shape
    gradient_magnitude_img = np.array((x_derivative_of_img), dtype = np.float64)
    for i in range(height):
        for j in range(width):
            gradient_magnitude_img[i][j] = math.sqrt(x_derivative_of_img[i][j]**2 + y_derivative_of_img[i][j]**2)

    return gradient_magnitude_img


cv2.imwrite("q4_noise_removed_sigma_5.png", x5_gaussian_filtering(noisy_image_sigma_5,5)) # 5x5 gaussian filtering on noisy_image with sigma 5
cv2.imwrite("q4_noise_removed_sigma_10.png", x5_gaussian_filtering(noisy_image_sigma_10,5))  # 5x5 gaussian filtering on noisy_image with sigma 10
cv2.imwrite("q4_noise_removed_sigma_20.png", x5_gaussian_filtering(noisy_image_sigma_20,5)) # 5x5 gaussian filtering on noisy_image with sigma 20

cv2.imwrite("q4_x_derivative_noise_removed_sigma_5.png", (x_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_5,5), 3))) # x derivative of noisy_image with sigma 5
cv2.imwrite("q4_y_derivative_noise_removed_sigma_5.png", (y_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_5,5), 3))) # y derivative of noisy_image with sigma 5

cv2.imwrite("q4_x_derivative_noise_removed_sigma_10.png", (x_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_10,5), 3))) # x derivative of noisy_image with sigma 10
cv2.imwrite("q4_y_derivative_noise_removed_sigma_10.png", (y_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_10,5), 3))) # y derivative of noisy_image with sigma 10

cv2.imwrite("q4_x_derivative_noise_removed_sigma_20.png", (x_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_20,5), 3))) # x derivative of noisy_image with sigma 20
cv2.imwrite("q4_y_derivative_noise_removed_sigma_20.png", (y_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_20,5), 3))) # y derivative of noisy_image with sigma 20


# find the gradient magnitude to see the gradiant magnitude of the noise removed images to see the effect better
cv2.imwrite("q4_noise_removed_gradient_magnitude_5.png", gradient_magnitude(      x_derivative_of_img_using_sobel(  x5_gaussian_filtering(noisy_image_sigma_5,5), 3 )     ,    y_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_5 ,5) ,  3  )    ) ) 
cv2.imwrite("q4_noise_removed_gradient_magnitude_10.png", gradient_magnitude(     x_derivative_of_img_using_sobel(  x5_gaussian_filtering(noisy_image_sigma_10,5), 3 )    ,    y_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_10,5) ,  3  )    ) )
cv2.imwrite("q4_noise_removed_gradient_magnitude_20.png", gradient_magnitude(     x_derivative_of_img_using_sobel(  x5_gaussian_filtering(noisy_image_sigma_20,5), 3 )    ,    y_derivative_of_img_using_sobel(x5_gaussian_filtering(noisy_image_sigma_20,5) ,  3  )    ) )


