'''
AUTHORS:
ERTURK OCAK 150200048
BATURALP INCE 150200738
'''

import numpy as np
import cv2
import math

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


def x_derivative_of_img_using_sobel(grayscaled_img,stride): # this function is explained in previous questions
    height, width = grayscaled_img.shape 
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # 
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


img_as_BGR = cv2.imread("darth_vader.png", cv2.IMREAD_COLOR) 
img_as_RGB = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.float64)
grayscaled_img = change_to_grayscale(img_as_RGB)

# astype is added for assertion error handling because float values have been used for the functions in this code previously
x_derivative_of_img = x_derivative_of_img_using_sobel(grayscaled_img, 3).astype(np.int16)  # x derivative of image
y_derivative_of_img = y_derivative_of_img_using_sobel(grayscaled_img, 3).astype(np.int16)  # y derivative of image
cv2.imwrite("q5_gradient_magnitude.png", gradient_magnitude(x_derivative_of_img, y_derivative_of_img)) # gradient magnitude of the image




x_derivative_of_img = x_derivative_of_img_using_sobel(grayscaled_img, 3).astype(np.int16) 
y_derivative_of_img = y_derivative_of_img_using_sobel(grayscaled_img, 3).astype(np.int16)

canny_image_1 = cv2.Canny(x_derivative_of_img, y_derivative_of_img, 60, 180) # Canny algorithm with low threshold 50 and high threshold 150
canny_image_2 = cv2.Canny(x_derivative_of_img, y_derivative_of_img, 175, 350) # Canny algorithm with low threshold 200 and high threshold 400

cv2.imwrite("q5_canny_1.png", canny_image_1) # Canny image 1
cv2.imwrite("q5_canny_2.png", canny_image_2) # Canny image 2 with different thresholds than canny image 1
