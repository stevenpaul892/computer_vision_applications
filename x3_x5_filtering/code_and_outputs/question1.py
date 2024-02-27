# Authors:
# Ertürk Ocak 150200048
# Baturalp İnce 150200738

import numpy as np
import cv2

img_as_BGR = cv2.imread("darth_vader.png", cv2.IMREAD_COLOR) 
img_as_RGB = np.array(cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB), dtype= np.uint8) 

def change_to_grayscale(img_as_RGB): # a function to change an RGB image to grayscaled version, same
                                    # implementation as in hw1
    
    height = img_as_RGB.shape[0] 
    width = img_as_RGB.shape[1] 
    grayscaled_img = np.full((height,width),0) 
    for i in range(height):
        for j in range(width):
            red = img_as_RGB[i][j][0] 
            green = img_as_RGB[i][j][1]  
            blue = img_as_RGB[i][j][2]
            grayscale_value = round(0.299 * red + 0.587 * green + 0.114 * blue) 
            grayscaled_img[i][j] = grayscale_value 
    return grayscaled_img


img_as_grayscaled = np.array(change_to_grayscale(img_as_RGB), dtype=np.uint8) # take the image as grayscaled



def zero_padding_grayscaled(grayscaled_img, padding): # zero padding for a grayscaled image
    stride = int((padding-1)/2) # stride taken
    number_of_rows = len(grayscaled_img)  # number of rows and columns of the image
    number_of_columns = len(grayscaled_img[0])

    holder_matrix = [  # holder image enlarged for the zero padding filled with all 0's
        [0 for col in range(number_of_columns + stride * 2)]
        for row in range(number_of_rows + stride * 2)
    ]

    for row in range(number_of_rows):
        for col in range(number_of_columns):
            holder_matrix[row + stride][col + stride] = grayscaled_img[row][col] # put the original image in the middle of the holder matrix

    return np.array(holder_matrix)  # return this matrix as numpy array

def zero_padding_RGB(rgb_img, padding): # zero padding for a RGB image
    stride = int((padding - 1) / 2) # stride taken
    height, width, channels = rgb_img.shape # take the number of rows and columns + the RGB channel of the image

    padded_height = height + 2 * stride # new height and width for the zero padded holder_matrix
    padded_width = width + 2 * stride

    holder_matrix = np.zeros((padded_height, padded_width, channels), dtype=np.uint8) # fill with zeros

    holder_matrix[stride:stride+height, stride:stride+width, :] = rgb_img # put the RGB image in the middle of the holder matrix

    return np.array(holder_matrix) # return this matrix as numpy array



def x3_filtering_grayscaled(grayscaled_img,stride): # 3x3 filtering for grayscaled image
    height = grayscaled_img.shape[0] # sizes of the image
    width = grayscaled_img.shape[1]

    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img, stride) # first zero pad the image so that the shape doesn't change
    x3_filtered_img = np.array((zero_padded_grayscaled_img), dtype = np.uint8) # this is the array to be returned

    mask = np.ones([3, 3], dtype =np.uint8 ) # a mask matrix size of 3x3 for interpolation
    mask = mask / 9 # the coefficient (stride*stride)

    k = int((stride-1)/2) # a stride for accessing the +k'th elements of height and width because of zero_padding

    for i in range(height): # for every element in the zero padded image (we will access +k'th elements to the right and above)
        for j in range(width): 
            # calculating the mean around the pixel
            mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            x3_filtered_img[i+k][j+k] = round(mean_around) # assign this interpolated  value to our array
    return x3_filtered_img



def x5_filtering_grayscaled(grayscaled_img,stride): # 5x5 filtering for grayscaled image
    height = grayscaled_img.shape[0] # sizes of the image
    width = grayscaled_img.shape[1]

    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img, stride) # first zero pad the image so that the shape doesn't change
    x5_filtered_img = np.array((zero_padded_grayscaled_img), dtype = np.uint8) # this is the array to be returned

    k = int((stride-1)/2) # a stride for accessing the +k'th elements of height and width because of zero_padding
    
    for i in range(height): # for every element in the zero padded image (we will access +k'th elements to the right and above)
        for j in range(width): 
            total_around = 0 # this total will be used for mean calculation
            for x in range(-k,k+1): # search around the pixel
                for m in range(-k,k+1): # search around the pixel
                    total_around += zero_padded_grayscaled_img[i+k+x][j+k+m] # sum up all the pixels around
            x5_filtered_img[i+k][j+k] = round(total_around/(stride*stride)) # divide the sum by the number of elements and assign it to the respective pixel
    return x5_filtered_img

def x3_filtering_RGB(RGB_img,stride): # 3x3 filtering for RGB image

    height, width, channels = RGB_img.shape # get the height, width and RGB channels of the image

    zero_padded_RGB_img = zero_padding_RGB(RGB_img, stride) # first zero pad the image so that the shape doesn't change
    x3_filtered_img = np.array((zero_padded_RGB_img), dtype = np.uint8) # this is the array to be returned

    mask = np.ones([3, 3], dtype =np.uint8) 
    mask = mask / 9
    
    k = int((stride-1)/2)

    for i in range(height):
        for j in range(width): 
            mean_around_R = 0 # calculation of mean around for RED channel
            mean_around_R += zero_padded_RGB_img[i+k-1, j+k-1,0]*mask[0, 0] + zero_padded_RGB_img[i+k-1, j+k,0]*mask[0, 1] + zero_padded_RGB_img[i+k-1, j+k + 1, 0]*mask[0, 2] + zero_padded_RGB_img[i+k, j+k-1, 0]*mask[1, 0] + zero_padded_RGB_img[i+k, j+k, 0]*mask[1, 1] + zero_padded_RGB_img[i+k, j+k + 1, 0]*mask[1, 2] + zero_padded_RGB_img[i+k + 1, j+k-1, 0]*mask[2, 0] + zero_padded_RGB_img[i+k + 1, j+k, 0]*mask[2, 1] +  zero_padded_RGB_img[i+k + 1, j+k + 1, 0]*mask[2, 2]
            x3_filtered_img[i+k][j+k][0] = round(mean_around_R)

            mean_around_G = 0 # calculation of mean around for GREEN channel
            mean_around_G += zero_padded_RGB_img[i+k-1, j+k-1,1]*mask[0, 0] + zero_padded_RGB_img[i+k-1, j+k,1]*mask[0, 1] + zero_padded_RGB_img[i+k-1, j+k + 1, 1]*mask[0, 2] + zero_padded_RGB_img[i+k, j+k-1, 1]*mask[1, 0] + zero_padded_RGB_img[i+k, j+k, 1]*mask[1, 1] + zero_padded_RGB_img[i+k, j+k + 1, 1]*mask[1, 2] + zero_padded_RGB_img[i+k + 1, j+k-1, 1]*mask[2, 0] + zero_padded_RGB_img[i+k + 1, j+k, 1]*mask[2, 1] +  zero_padded_RGB_img[i+k + 1, j+k + 1, 1]*mask[2, 2]
            x3_filtered_img[i+k][j+k][1] = round(mean_around_G)

            mean_around_B = 0 # calculation of mean around for BLUE channel
            mean_around_B += zero_padded_RGB_img[i+k-1, j+k-1,2]*mask[0, 0] + zero_padded_RGB_img[i+k-1, j+k,2]*mask[0, 1] + zero_padded_RGB_img[i+k-1, j+k + 1, 2]*mask[0, 2] + zero_padded_RGB_img[i+k, j+k-1, 2]*mask[1, 0] + zero_padded_RGB_img[i+k, j+k, 2]*mask[1, 1] + zero_padded_RGB_img[i+k, j+k + 1, 2]*mask[1, 2] + zero_padded_RGB_img[i+k + 1, j+k-1, 2]*mask[2, 0] + zero_padded_RGB_img[i+k + 1, j+k, 2]*mask[2, 1] +  zero_padded_RGB_img[i+k + 1, j+k + 1, 2]*mask[2, 2]
            x3_filtered_img[i+k][j+k][2] = round(mean_around_B)
    return x3_filtered_img

def x5_filtering_RGB(RGB_img,stride): # 5x5 filtering for RGB image

    height, width, channels = RGB_img.shape # get the height, width and RGB channels of the image

    zero_padded_RGB_img = zero_padding_RGB(RGB_img, stride) # first zero pad the image so that the shape doesn't change
    x5_filtered_img = np.array((zero_padded_RGB_img), dtype = np.uint8) # this is the array to be returned

    k= int((stride-1)/2)

    for i in range(height):
        for j in range(width): 
            # for every iteration of pixels, we will refresh total around R,G,B to 0
            total_around_R = 0
            total_around_G = 0
            total_around_B = 0
            for x in range(-k,k+1):
                for m in range(-k,k+1):
                    # calculate the sum around for each channel
                    total_around_R += zero_padded_RGB_img[i+k+x][j+k+m][0]
                    total_around_G += zero_padded_RGB_img[i+k+x][j+k+m][1]
                    total_around_B += zero_padded_RGB_img[i+k+x][j+k+m][2]
            # divide the total by the number of elements around and assign it to respective channels of the pixel
            x5_filtered_img[i+k][j+k][0] = round(total_around_R/25)
            x5_filtered_img[i+k][j+k][1] = round(total_around_G/25)
            x5_filtered_img[i+k][j+k][2] = round(total_around_B/25)

    return x5_filtered_img

cv2.imwrite("darth_vader_grayscaled.png", img_as_grayscaled) 

np.save("q1_x3_filtering_grayscaled.npy", x3_filtering_grayscaled(img_as_grayscaled,3))
cv2.imwrite("x3_filtering_grayscaled.png", x3_filtering_grayscaled(img_as_grayscaled,3)) 

np.save("q1_x5_filtering_grayscaled.npy", x5_filtering_grayscaled(img_as_grayscaled,5))
cv2.imwrite("x5_filtering_grayscaled.png", x5_filtering_grayscaled(img_as_grayscaled,5)) 

np.save("q1_x3_filtering_RGB.npy", x3_filtering_RGB(img_as_RGB,3))
cv2.imwrite("x3_filtering_RGB.png", x3_filtering_RGB(img_as_RGB,3))

np.save("q1_x5_filtering_RGB.npy", x5_filtering_RGB(img_as_RGB,5))
cv2.imwrite("x5_filtering_RGB.png", x5_filtering_RGB(img_as_RGB,5))