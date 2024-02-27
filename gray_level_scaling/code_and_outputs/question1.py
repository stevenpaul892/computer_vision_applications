# Authors:
# Ertürk Ocak 150200048
# Baturalp İnce 150200738

import cv2 
import numpy as np


img_as_BGR = cv2.imread("tiesto.png", cv2.IMREAD_COLOR) # read the image as BGR numpy array
img_as_RGB = cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB) # change the BGR image to RGB


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


np.save("q1_RGB_img.npy",img_as_RGB) # saving the input image as .npy file
np.save("q1_grayscaled_img.npy", change_to_grayscale(img_as_RGB)) # saving the output image as .npy file

cv2.imwrite("grayscaled_tiesto.png", change_to_grayscale(img_as_RGB)) # saving the output image as .png file for the report
