'''
AUTHORS:
ERTURK OCAK 150200048
BATURALP INCE 150200738
'''


import numpy as np
import cv2

img_as_BGR = cv2.imread("darth_vader.png", cv2.IMREAD_COLOR) # reading the image as BGR
img_as_RGB = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.uint8) # then converting it to RGB

def change_to_grayscale(img_as_RGB): # this is the same function used in the homeworks before
    
    height = img_as_RGB.shape[0] 
    width = img_as_RGB.shape[1] 
    grayscaled_img = np.zeros((height,width), dtype= np.uint8) 
    for i in range(height):
        for j in range(width):
            red = img_as_RGB[i][j][0] 
            green = img_as_RGB[i][j][1]  
            blue = img_as_RGB[i][j][2]
            grayscale_value = round(0.299 * red + 0.587 * green + 0.114 * blue) 
            grayscaled_img[i][j] = grayscale_value 
    return grayscaled_img


def zero_padding_grayscaled(grayscaled_img, padding):  #this is the same zero padding used in the homeworks before
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


def x3_gaussian_filtering(RGB_img,stride):
    grayscaled_img = change_to_grayscale(RGB_img) # grayscale the image first 
    height = grayscaled_img.shape[0] # get the sizes
    width = grayscaled_img.shape[1]
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # zero pad the image so that image size won't change
    gaussian_filtered_img = np.array((zero_padded_grayscaled_img), dtype= np.uint8) # create an array to be filled in with the correct values

    mask = np.array([[1,2,1], # this is the gaussian filter to be used in this 3x3 filtering
                     [2,4,2],
                     [1,2,1]])
    mask = mask / 16 # this is the dividing factor for the mask (total of 16 inside matrix)

    k = int((stride-1)/2) # this k value will be used to access to the correct positions in the zero padded image
    for i in range (height):
        for j in range (width):
            # take the weighted mean around with the given shape using the gaussian filter
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            gaussian_filtered_img[i+k][j+k] = round(weighted_mean_around)
    
    return gaussian_filtered_img

def x5_gaussian_filtering(RGB_img, stride):
    grayscaled_img = change_to_grayscale(RGB_img) # take the image as grayscaled first
    height, width = grayscaled_img.shape # sizes
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride) # zero pad the image so that image size won't change
    gaussian_filtered_img = np.array((zero_padded_grayscaled_img), dtype= np.uint8) # create an array to be filled in with the correct values

    # this is the 5x5 gaussian filter to be used
    mask = np.array([[1, 4, 6, 4, 1], 
                     [4, 16, 24, 16, 4],
                     [6, 24, 36, 24, 6],
                     [4, 16, 24, 16, 4],
                     [1, 4, 6, 4, 1]])
    mask = mask / 256 # this is the dividing factor for the mask (total of 256 inside matrix)

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = (
                        # take the weighted mean around with the given shape using the gaussian filter
                        # by accessing each pixel and its neighbours one by one
                        zero_padded_grayscaled_img[i + k, j + k - 2] * mask[2, 0]
                        + zero_padded_grayscaled_img[i + k, j + k - 1] * mask[2, 1]
                        + zero_padded_grayscaled_img[i + k, j + k] * mask[2, 2]
                        + zero_padded_grayscaled_img[i + k, j + k + 1] * mask[2, 3]
                        + zero_padded_grayscaled_img[i + k, j + k + 2] * mask[2, 4]
                        + zero_padded_grayscaled_img[i + k - 2, j + k - 2] * mask[0, 0]
                        + zero_padded_grayscaled_img[i + k - 2, j + k - 1] * mask[0, 1]
                        + zero_padded_grayscaled_img[i + k - 2, j + k] * mask[0, 2]
                        + zero_padded_grayscaled_img[i + k - 2, j + k + 1] * mask[0, 3]
                        + zero_padded_grayscaled_img[i + k - 2, j + k + 2] * mask[0, 4]
                        + zero_padded_grayscaled_img[i + k - 1, j + k - 2] * mask[1, 0]
                        + zero_padded_grayscaled_img[i + k - 1, j + k - 1] * mask[1, 1]
                        + zero_padded_grayscaled_img[i + k - 1, j + k] * mask[1, 2]
                        + zero_padded_grayscaled_img[i + k - 1, j + k + 1] * mask[1, 3]
                        + zero_padded_grayscaled_img[i + k - 1, j + k + 2] * mask[1, 4]
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

cv2.imwrite("darth_vader_grayscaled.png", change_to_grayscale(img_as_RGB)) # original image
cv2.imwrite("q1_x3_gaussian_filtered_img.png", x3_gaussian_filtering(img_as_RGB,3)) # 3x3 gaussian filtered image
cv2.imwrite("q1_x5_gaussian_filtered_img.png", x5_gaussian_filtering(img_as_RGB,5)) # 5x5 gaussian filtered image