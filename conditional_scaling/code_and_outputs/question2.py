# Authors:
# Ertürk Ocak 150200048
# Baturalp İnce 150200738

import cv2
import numpy as np

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

def calculate_mean(image):
    img_grayscaled = change_to_grayscale(image) # holder image for accessing the elements
    height = img_grayscaled.shape[0]
    width = img_grayscaled.shape[1]
    sum = 0
    for i in range(height):
        for j in range(width):
            sum = sum + (img_grayscaled[i][j])# mean calculation
    return sum/(height*width) # mean calculation
 

def calculate_variance(image):
    img_grayscaled = change_to_grayscale(image) # holder image for accessing the elements
    height = img_grayscaled.shape[0]
    width = img_grayscaled.shape[1]
    sum = 0
    mean = calculate_mean(image)
    
    for i in range(height):
        for j in range(width):
            sum = sum + (img_grayscaled[i][j]- mean)**2 # variance calculation
    return sum/(height*width) # variance calculation


main_img = cv2.cvtColor(cv2.imread("tomjerry.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) # read the original image
main_mean = calculate_mean(main_img)
main_variance = calculate_variance(main_img)

reference_img = cv2.cvtColor(cv2.imread("matrix.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) # read the reference image
reference_mean = calculate_mean(reference_img)
reference_variance = calculate_variance(reference_img)

def intensify(main_image,main_mean,main_variance,reference_mean,reference_variance):
    grayscaled_img = change_to_grayscale(main_image) # change the original image to grayscaled first
    height = grayscaled_img.shape[0] # get the sizes of the image to access the pixels one by one
    width = grayscaled_img.shape[1]
    for i in range(height):
        for j in range(width):
            intensified_value = round(reference_mean * (main_variance/reference_variance)-main_mean + (grayscaled_img[i][j]))*reference_variance/main_variance # the formula
            grayscaled_img[i][j] = intensified_value # write the new number to our img 

    return grayscaled_img # new grayscaled image having same mean and variance as the reference image


np.save("q2_RGB_img_tomjerry.npy",main_img) # save the original image as .npy file
np.save("q2_RGB_img_matrix.npy",reference_img) # save the original image as .npy file
cv2.imwrite("grayscaled_tomjerry.png", change_to_grayscale(main_img)) # saving the output image as .png file for the reprot
np.save("q2_CondScaled_Grayscale_img.npy", intensify(main_img,main_mean, main_variance, reference_mean, reference_variance)) # save the intensified image with respect to reference image as .npy file
np.save("q2_grayscaled_matrix.npy",change_to_grayscale(reference_img)) # save the original image as .npy file

cv2.imwrite("CondScaled_grayscale_img.png", intensify(main_img,main_mean, main_variance, reference_mean, reference_variance)) # saving the output image as .png file for the report
cv2.imwrite("grayscaled_matrix_img.png", change_to_grayscale(reference_img)) # saving the output image as .png file for the report

