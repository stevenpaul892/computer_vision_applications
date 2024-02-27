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
    grayscaled_img = np.full((height,width),0) 
    for i in range(height):
        for j in range(width):
            red = img_as_RGB[i][j][0] 
            green = img_as_RGB[i][j][1]  
            blue = img_as_RGB[i][j][2]
            grayscale_value = round(0.299 * red + 0.587 * green + 0.114 * blue) 
            grayscaled_img[i][j] = grayscale_value 
    return grayscaled_img

img_as_grayscaled = change_to_grayscale(img_as_RGB).astype(np.uint8)


def zero_padding_grayscaled(grayscaled_img, padding): # same zero padding function which was commented in the first question (for grayscaled image in this question)
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

def x3_filtering_grayscaled(grayscaled_img,stride): # this is the same function which was used in 1st question, so i will not comment it out again.
    height = grayscaled_img.shape[0]
    width = grayscaled_img.shape[1]

    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img, stride)
    x3_filtered_img = np.array((zero_padded_grayscaled_img), dtype = np.uint8)

    mask = np.ones([3, 3], dtype =np.uint8 ) 
    mask = mask / 9

    k = int((stride-1)/2)

    for i in range(height):
        for j in range(width): 
            mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            x3_filtered_img[i+k][j+k] = round(mean_around)
    return x3_filtered_img

def downsample_image(img_as_grayscaled, factor):
    filtered_grayscale_image = x3_filtering_grayscaled(img_as_grayscaled,3) # first, linearly interpolate the grayscaled image so that when downsampled, the image will
                                                                            # look better (because when we assign pixel from the grayscaled original image, we will skip
                                                                            # every 2nd pixel, so we want the assigned pixels to represent the mean of its pixels around)
    height,width = img_as_grayscaled.shape # get the height and width of the image
    downsampled_width = width // factor # divide the width by 2
    downsampled_height = height // factor # divide the height by 2
    downsampled_image = np.zeros((downsampled_height, downsampled_width), dtype=np.uint8) # create an array for the downsampled size filled with 0's

    for x in range(0, height-1, 2): # for every 2nd pixel of original image from the start
        for y in range(0, width-1, 2):
            downsampled_image[x // 2][y // 2] =  filtered_grayscale_image[x][y] # assign the interpolated pixel of grayscaled image to the downsampled array
    return downsampled_image

cv2.imwrite("darth_vader_grayscaled.png", img_as_grayscaled)

np.save("q3_downsampled_img.npy", downsample_image(img_as_grayscaled,2))
cv2.imwrite("downsampled_img.png", downsample_image(img_as_grayscaled,2))