'''
AUTHORS:
ERTURK OCAK 150200048
BATURALP INCE 150200738
'''

import numpy as np
import cv2
import math


img_as_BGR = cv2.imread("darth_vader.png", cv2.IMREAD_COLOR) # reading the image as BGR
img_as_RGB = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.uint8) # then converting it to RGB

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


def zero_padding_grayscaled(grayscaled_img, padding):  # this function is explained in previous questions
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

def add_noise_to_image(img_as_RGB, sigma):

    grayscaled_img = change_to_grayscale(img_as_RGB) # first grayscale the RGB image
    height,width = grayscaled_img.shape # then get the sizes of the image
    
    noisy_img = np.array((grayscaled_img), dtype = np.float64) # create an array for the values to be filled in

    for i in range(height):
        for j in range(width): # for every pixel of the grayscaled image
          
            noisy_img[i][j] = noisy_img[i][j] + np.random.normal(0,sigma) # add noise as a random gaussian distributed sample using the function
                                                                        # with the parameter 0 to specify that it can add a random negative or
                                                                        # positive value as center is 0, and sigma parameter to specify the standart
                                                                        # derivation of the distribution from the center.
        
    return noisy_img


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

cv2.imwrite("darth_vader_grayscaled.png", change_to_grayscale(img_as_RGB))
cv2.imwrite("q3_noisy_img_sigma_5.png", add_noise_to_image(img_as_RGB,5))
cv2.imwrite("q3_noisy_img_sigma_10.png", add_noise_to_image(img_as_RGB,10))
cv2.imwrite("q3_noisy_img_sigma_20.png", add_noise_to_image(img_as_RGB,20))

cv2.imwrite("q3_x_derivative_of_img_sigma_5.png", x_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,5),3))
cv2.imwrite("q3_x_derivative_of_img_sigma_10.png", x_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,10),3))
cv2.imwrite("q3_x_derivative_of_img_sigma_20.png", x_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,20),3))

cv2.imwrite("q3_y_derivative_of_img_sigma_5.png", y_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,5),3))
cv2.imwrite("q3_y_derivative_of_img_sigma_10.png", y_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,10),3))
cv2.imwrite("q3_y_derivative_of_img_sigma_20.png", y_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,20),3))

cv2.imwrite("q3_gradient_magnitude_2d_sigma_5.png", gradient_magnitude(x_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,5),3), y_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,5),3)))
cv2.imwrite("q3_gradient_magnitude_2d_sigma_10.png", gradient_magnitude(x_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,10),3), y_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,10),3)))
cv2.imwrite("q3_gradient_magnitude_2d_sigma_20.png", gradient_magnitude(x_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,20),3), y_derivative_of_img_using_sobel(add_noise_to_image(img_as_RGB,20),3)))