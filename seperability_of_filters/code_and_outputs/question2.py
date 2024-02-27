'''
AUTHORS:
ERTURK OCAK 150200048
BATURALP INCE 150200738
'''


import numpy as np
import cv2
import math

img_as_BGR = cv2.imread("darth_vader.png", cv2.IMREAD_COLOR)  # reading the image as BGR
img_as_RGB = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.float64) # then converting it to RGB

def change_to_grayscale(img_as_RGB):
    
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

def zero_padding_grayscaled(grayscaled_img, padding): 
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

    return np.array(holder_matrix, dtype = np.float64)  


def x_derivative_of_img_using_sobel(img_as_RGB,stride):
    grayscaled_img = change_to_grayscale(img_as_RGB) # grayscale the image first
    height, width = grayscaled_img.shape # get the sizes
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # zero pad the image so that image size won't change
    x_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64) # create an array to be filled in with the suitable values
    mask = np.array([[1,2,1], # this is the mask to find x derivative of an image
                     [0,0,0],
                     [-1,-2,-1]])

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            # convolute the image with the given x derivative mask
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            x_derivative_of_img[i+k][j+k] = weighted_mean_around
    
    return x_derivative_of_img


def x_derivative_of_img_using_seperability(img_as_RGB,stride):
    grayscaled_img = change_to_grayscale(img_as_RGB) # grayscale the image first
    height, width = grayscaled_img.shape # get the sizes
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # zero pad the image so that image size won't change 
    x_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64) # create an array to be filled in with the suitable values


    mask_1 = np.array([1,2,1]) # this is the horizontal matrix to be convolved with the image first (horizontal access will be done in for loops)
    mask_2 = np.array([1,0,-1]) # this is the vertical matrix to be convolved with the image first (vertical access will be done in for loops)


    k = int((stride-1)/2) # the offset k to access the correct positions of the zero padded image

    for i in range (height):
        for j in range (width):
            # get the weighted mean of pixels on horizontal axis
            horizontal_convolution = zero_padded_grayscaled_img[i+k][j+k] * mask_1[1] + zero_padded_grayscaled_img[i+k][j+k+1] * mask_1[2] + zero_padded_grayscaled_img[i+k][j+k-1] * mask_1[0]
            x_derivative_of_img[i+k][j+k] = round(horizontal_convolution*3) # multiplication with 3 is done to show the same brightness with the normal x derivation

    for i in range (height):
        for j in range (width):
            # then, get the weighted mean of pixels on vertical axis
            vertical_convolution = zero_padded_grayscaled_img[i+k][j+k] * mask_2[1] + zero_padded_grayscaled_img[i+k+1][j+k] * mask_2[2] + zero_padded_grayscaled_img[i+k-1][j+k] * mask_2[0]
            x_derivative_of_img[i+k][j+k] = round(vertical_convolution*3) # multiplication with 3 is done to show the same brightness with the normal x derivation

    return x_derivative_of_img



def y_derivative_of_img_using_sobel(img_as_RGB,stride):
    grayscaled_img = change_to_grayscale(img_as_RGB) # grayscale the image first
    height, width = grayscaled_img.shape # then get the sizes
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # zero pad the image so that image size won't change
    y_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64) # create an array for the values to be filled in

    mask = np.array([[1,0,-1], # this is the matrix used for finding the y derivative of the image
                     [2,0,-2], 
                     [1,0,-1]])

    k = int((stride-1)/2) # k offset
    for i in range (height):
        for j in range (width):
            # convolute the image with the given y derivative mask
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            y_derivative_of_img[i+k][j+k] = weighted_mean_around
    
    return y_derivative_of_img

def y_derivative_of_img_using_seperability(img_as_RGB,stride):
    grayscaled_img = change_to_grayscale(img_as_RGB) # grayscale the image first
    height, width = grayscaled_img.shape # get the sizes
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # zero pad the image so that image size won't change
    y_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64) # create an array for the values to be filled in

    mask_1 = np.array([1,0,-1]) # this is the horizontal matrix to be convolved with the image first (horizontal access will be done in for loops)
    mask_2 = np.array([1,2,1]) # this is the vertical matrix to be convolved with the image first (vertical access will be done in for loops)

    k = int((stride-1)/2)

    for i in range (height):
        for j in range (width):
            # get the weighted mean of pixels on vertical axis
            weighted_mean_around = zero_padded_grayscaled_img[i+k][j+k] * mask_2[1] + zero_padded_grayscaled_img[i+k+1][j+k] * mask_2[2] + zero_padded_grayscaled_img[i+k+1][j+k] * mask_2[0]
            y_derivative_of_img[i+k][j+k] = round(weighted_mean_around*3)
    for i in range (height):
        for j in range (width):
            # then, get the weighted mean of pixels on horizontal axis
            weighted_mean_around = zero_padded_grayscaled_img[i+k][j+k] * mask_1[1] + zero_padded_grayscaled_img[i+k][j+k+1] * mask_1[2] + zero_padded_grayscaled_img[i+k][j+k-1] * mask_1[0]
            y_derivative_of_img[i+k][j+k] = round(weighted_mean_around*3)

    return y_derivative_of_img

def gradient_magnitude(x_derivative_of_img, y_derivative_of_img):
    height,width = x_derivative_of_img.shape # size any of the given 2 inputs to create a new array 
    gradient_magnitude_img = np.array((x_derivative_of_img), dtype = np.float64) # creating a new array for the values to be filled in
    for i in range(height):
        for j in range(width):
            # take the square root of the squares of each pixel for x derivative and y derivative
            gradient_magnitude_img[i][j] = math.sqrt(x_derivative_of_img[i][j]**2 + y_derivative_of_img[i][j]**2)

    return gradient_magnitude_img


cv2.imwrite("darth_vader_grayscaled.png", change_to_grayscale(img_as_RGB)) # grayscaled img
cv2.imwrite("q2_x_derivative_of_img.png", x_derivative_of_img_using_sobel(img_as_RGB,3)) # x derivative of img
cv2.imwrite("q2_y_derivative_of_img.png", y_derivative_of_img_using_sobel(img_as_RGB,3)) # y derivative of img
cv2.imwrite("q2_x_derivative_of_img_seperability.png", x_derivative_of_img_using_seperability(img_as_RGB,3)) # x derivative using 2 matrices created from a seperable matrix
cv2.imwrite("q2_y_derivative_of_img_seperability.png", y_derivative_of_img_using_seperability(img_as_RGB,3)) # y derivative using 2 matrices created from a seperable matrix
cv2.imwrite("q2_gradient_magnitude_2d.png", gradient_magnitude(x_derivative_of_img_using_sobel(img_as_RGB,3), y_derivative_of_img_using_sobel(img_as_RGB,3))) # gradient magnitude of the image
cv2.imwrite("q2_gradient_magnitude_seperability.png", gradient_magnitude(x_derivative_of_img_using_seperability(img_as_RGB,3), y_derivative_of_img_using_seperability(img_as_RGB,3))) # gradient magnitude of the image using seperability 