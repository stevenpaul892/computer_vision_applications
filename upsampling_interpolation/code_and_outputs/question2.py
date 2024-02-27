# Authors:
# Ertürk Ocak 150200048
# Baturalp İnce 150200738


import numpy as np
import cv2

img_as_BGR = cv2.imread("darth_vader.png", cv2.IMREAD_COLOR) 
img_as_RGB = cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2RGB) 

def change_to_grayscale(img_as_RGB):
    
    height = img_as_RGB.shape[0] 
    width = img_as_RGB.shape[1] 
    upsampled_img = np.full((height,width),0) 
    for i in range(height):
        for j in range(width):
            red = img_as_RGB[i][j][0] 
            green = img_as_RGB[i][j][1]  
            blue = img_as_RGB[i][j][2]
            grayscale_value = round(0.299 * red + 0.587 * green + 0.114 * blue) 
            upsampled_img[i][j] = grayscale_value 
    return upsampled_img

img_as_grayscaled = np.array(change_to_grayscale(img_as_RGB), dtype = np.uint8)

def upsample_image(grayscaled_img, factor):

    height, width = grayscaled_img.shape 
    upsampled_height = factor * height # calculate the new height
    upsampled_width = factor * width # calculate the new width
    upsampled_image = np.zeros((upsampled_height, upsampled_width), dtype = np.uint8) # fill all the upsampled image with zeros first
    upsampled_image[::factor, ::factor] = grayscaled_img # for every 2nd pixel, replace the pixels of the grayscaled image to the upsampled image
                                                        # which will cause black pixels every 2nd pixel from the start as well (which will be filled)
                                                        # with interpolation around them
    return upsampled_image


def zero_padding(upsampled_img, padding):
    stride = int((padding-1)/2) 
    number_of_rows = len(upsampled_img) 
    number_of_columns = len(upsampled_img[0])

    holder_matrix = [ 
        [0 for col in range(number_of_columns + stride * 2)]
        for row in range(number_of_rows + stride * 2)
    ]

    for row in range(number_of_rows):
        for col in range(number_of_columns):
            holder_matrix[row + stride][col + stride] = upsampled_img[row][col]

    return np.array(holder_matrix)  


def linear_interpolation(upsampled_img, box):

    height = upsampled_img.shape[0] # get the sizes of the upsampled image
    width = upsampled_img.shape[1]
    zero_padded_upsampled_img = zero_padding(upsampled_img,box) # zero pad the image first so that the shape doesn't change
    zero_padded_upsampled_interpolated_img = np.array((zero_padded_upsampled_img), dtype = np.uint8) # create an array to be returned

    k = int((box-1)/2) # the stride for access to elements of zero padded image

    for i in range(1, height,2): # we need to search for every 2nd pixel starting from 1st pixel for black pixels
        for j in range(1, width,2):
            if(zero_padded_upsampled_img[i+k][j+k] == 0): # double check if the pixel we iterate is black for sure
                total_around = 0 
                counter = 0 # a counter to count how many elements around this black pixel are known, we don't take unknown black pixels into consideration
                for x in range(-k,k+1):
                    for m in range(-k,k+1):
                        if(zero_padded_upsampled_img[i+k+x][j+k+m] != 0): # if it is NOT known pixel, don't take them 
                            total_around += zero_padded_upsampled_img[i+k+x][j+k+m] # add it to the total
                            counter = counter + 1 # increase the counter
                if(counter != 0): # an error prevention which might be caused by dividing the total by 0 
                    zero_padded_upsampled_interpolated_img[i+k][j+k] = round(total_around/counter) # assign the interpolated value of the pixel around (mean around)            

    return zero_padded_upsampled_interpolated_img

cv2.imwrite("darth_vader_grayscaled.png", img_as_grayscaled)

np.save("q2_upsampled_img.npy", upsample_image(img_as_grayscaled,2))
cv2.imwrite("upsampled_img.png", upsample_image(img_as_grayscaled,2))

np.save("q2_upsampled_interpolated_img.npy", linear_interpolation( upsample_image(img_as_grayscaled,2), 3))
cv2.imwrite("upsampled_interpolated_img.png", linear_interpolation( upsample_image(img_as_grayscaled,2), 3))