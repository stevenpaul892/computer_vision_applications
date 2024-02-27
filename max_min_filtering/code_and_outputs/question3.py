# Authors:
# Ertürk Ocak 150200048
# Baturalp İnce 150200738

import cv2 
import numpy as np


img = cv2.cvtColor(cv2.imread("dinosaur.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) # read the image as RGB

def change_to_grayscale(img_as_RGB):
    
    height = img_as_RGB.shape[0] # image sizes
    width = img_as_RGB.shape[1] # image sizes
    grayscaled_img = np.full((height,width),0) # I am using np.zeros only to create an empty array, no other use. (It was causing a lot of problems to not use it, so I had to)
                                                # I couldn't find any other way then creating a new numpy array, because if i assign img_as_RGB directly to it, it was taking 3rd
                                                # dimension as well, so I couldn't find any method to take out a dimension without using numpy functions anyways. So I just used it
                                                # at the beginning creating an numpy array instead of dealing with a lot of numpy functions.

    for i in range(height):
        for j in range(width):
            red = img_as_RGB[i][j][0] # get the RGB channels for each of the pixel in the image
            green = img_as_RGB[i][j][1]  
            blue = img_as_RGB[i][j][2]
            grayscale_value = round(0.299 * red + 0.587 * green + 0.114 * blue) # this function is to find the grayscale value of the pixel
            grayscaled_img[i][j] = grayscale_value # put this value into the pixel again as one of 0-255 values
    return grayscaled_img

def zero_padding(RGB_img, padding):
    grayscaled_img = change_to_grayscale(RGB_img) # change the image to grayscale first
    stride = int((padding-1)/2) # how many rows&columns we need to search more for each corner pixel
    number_of_rows = len(grayscaled_img) # get the number of rows and columns
    number_of_columns = len(grayscaled_img[0])

 
    holder_matrix = [ # creating a new matrix full of zeros with the given stride size
        [0 for col in range(number_of_columns + stride * 2)]
        for row in range(number_of_rows + stride * 2)
    ]

    for row in range(number_of_rows): # the elements in our original matrix will be added inside to our new holder matrix
        for col in range(number_of_columns):
            holder_matrix[row + stride][col + stride] = grayscaled_img[row][col]

    return np.array(holder_matrix)  # change the matrix to numpy array for image processing use


def maximum_filtering(img,stride):
    height = img.shape[0] # get the sizes of the original RGB image
    width = img.shape[1]

    zeropadded_img = zero_padding(img,stride) # create a new zero_padded numpy array for the local operations

    max_filtered_img = np.array((zeropadded_img)) # copy the same image to another numpy array for the local image process

    k = int((stride-1)/2) # this is to store the number of zero's added to each side
    for i in range(height):
        for j in range(width):
            max = zeropadded_img[i+k][j+k] # if we don't find any number bigger than this, we will keep it same
            for x in range(-k,k+1):
                for m in range(-k,k+1):
                    if zeropadded_img[i+k+x][j+k+m] > max: # check the local pixels for the maximum value
                        max = zeropadded_img[i+k+x][j+k+m]
            max_filtered_img[i+k][j+k] = max
    return max_filtered_img

def minimum_filtering(img,stride):
    height = img.shape[0] # get the sizes of the original RGB image
    width = img.shape[1]
    zeropadded_img = zero_padding(img,stride) # create a new zero_padded numpy array for the local operations
    min_filtered_img = np.array((zeropadded_img)) # copy the same image to another numpy array for the local image process
    k = int((stride-1)/2) # this is to store the number of zero's added to each side    
    for i in range(height):
        for j in range(width):
            min = zeropadded_img[i+k][j+k] # if we don't find any number smaller than this, we will keep it same
            for x in range(-k,k+1):
                for m in range(-k,k+1):
                    if zeropadded_img[i+k+x][j+k+m] < min: # check the local pixels for the minimum value
                        min = zeropadded_img[i+k+x][j+k+m]
            min_filtered_img[i+k][j+k] = min

    return min_filtered_img

np.save("q3_RGB_img.npy",img)  # save the original image as .npy file
np.save("q3_grayscaled_img.npy",change_to_grayscale(img)) # this and all below are needed .npy files

np.save("q3_7_zeropadding_img.npy",zero_padding(img,7))     
np.save("q3_21_zeropadding_img.npy",zero_padding(img,21)) 

np.save("q3_7_maxfilt_img.npy",maximum_filtering(img,7)) 
np.save("q3_21_maxfilt_img.npy",maximum_filtering(img,21)) 

np.save("q3_7_minfilt_img.npy",minimum_filtering(img,7)) 
np.save("q3_21_minfilt_img.npy",minimum_filtering(img,21))


cv2.imwrite("dinosaur_grayscaled.png", change_to_grayscale(img)) # saving the output image as .png file for the report

cv2.imwrite("7_zeropadding_img.png", zero_padding(img,7)) # saving the output image as .png file for the report
cv2.imwrite("21_zeropadding_img.png", zero_padding(img,21)) # saving the output image as .png file for the report

cv2.imwrite("7_maxfilt.png",maximum_filtering(img,7))  # saving the output image as .png file for the report
cv2.imwrite("21_maxfilt.png",maximum_filtering(img,21))  # saving the output image as .png file for the report

cv2.imwrite("7_minfilt.png",minimum_filtering(img,7))  # saving the output image as .png file for the report
cv2.imwrite("21_minfilt.png",minimum_filtering(img,21))  # saving the output image as .png file for the report