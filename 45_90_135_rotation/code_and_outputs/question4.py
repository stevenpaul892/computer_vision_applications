# Authors:
# Ertürk Ocak 150200048
# Baturalp İnce 150200738


import numpy as np
import cv2
import math

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


img_as_grayscaled = np.array(change_to_grayscale(img_as_RGB), dtype = np.uint8)



def rotated_grayscale_img(grayscaled_img, degree):
   
    radians = math.radians(degree) # first we have to convert from degree to radians for sin/cos functions
    
    new_height = round(abs(grayscaled_img.shape[0]*math.cos(radians))) + round(abs(grayscaled_img.shape[1]*math.sin(radians)))
    # this new height is calculated considering the angle to be rotated and the height/width of the original image (round needed to convert it to integer)
        
    new_width = round(abs(grayscaled_img.shape[1]*math.cos(radians))) + round(abs(grayscaled_img.shape[0]*math.sin(radians)))
    # this new height is calculated considering the angle to be rotated and the height/width of the original image (round needed to convert it to integer)
   
    # since we rotate with respect to the center, we have to calculate the center of the original image which will be used later
    mid_x_old, mid_y_old = (grayscaled_img.shape[1]//2, grayscaled_img.shape[0]//2)

    # since we rotate with respect to the center, we have to calculate the center of the rotated image which will be used later  
    mid_x_new,mid_y_new = (new_height//2, new_width//2)
        
    # create an empty array filled with zeros with the size of new_height and new_width
    rotated_img = np.zeros((new_height,new_width), dtype= np.uint8)

    for i in range(rotated_img.shape[0]): # for every pixel of the rotated image
        for j in range(rotated_img.shape[1]):

            # for calculations of x_old, y_old; coordinates of the center of rotated image must be subtracted because we rotate with respect to center
            x_old= (i-mid_x_new)*math.cos(radians)+(j-mid_y_new)*math.sin(radians) # find the location of the x coordinate of the pixel in the original image
            y_old= -(i-mid_x_new)*math.sin(radians)+(j-mid_y_new)*math.cos(radians) # find the location of the y coordinate of the pixel in the original image
            
            x_old=round(x_old)+mid_y_old # now we have to slide/shift the coordinates found as much as the location of the center of the old image
                                        # since the rotation is completed
            y_old=round(y_old)+mid_x_old # we have to round x_old and y_old just in case we get non-integer number

            # this check is to validate that the coordinates we found for original image actually exist and have an intense value 
            if (x_old>=0 and y_old>=0 and x_old<grayscaled_img.shape[0] and  y_old<grayscaled_img.shape[1]): 
                rotated_img[i,j] = grayscaled_img[x_old,y_old] # now that we checked it does exist, we can assign that pixel for the suitable location in the 
                                                                # rotated image

    return rotated_img 

def x3_filtering_grayscaled(grayscaled_img,stride): # same 3x3 filtering function which was commented before
    height = grayscaled_img.shape[0]
    width = grayscaled_img.shape[1]

    zero_padded_grayscaled_img = zero_padding(grayscaled_img, stride)
    x3_filtered_img = np.array((zero_padded_grayscaled_img), dtype = np.uint8)

    mask = np.ones([3, 3], dtype =np.uint8 ) 
    mask = mask / 9

    k = int((stride-1)/2)

    for i in range(height):
        for j in range(width): 
            mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            x3_filtered_img[i+k][j+k] = round(mean_around)
    return x3_filtered_img


def zero_padding(grayscaled_img, padding): # same zero padding(grayscaled) function which was commented before
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

cv2.imwrite("darth_vader_grayscaled.png", img_as_grayscaled)

np.save("q4_45_degree_rotation.npy", rotated_grayscale_img(img_as_grayscaled,45))
cv2.imwrite("45_degree_rotation.png", rotated_grayscale_img(img_as_grayscaled,45))

np.save("q4_45_degree_rotation_interpolated.npy", x3_filtering_grayscaled(rotated_grayscale_img(img_as_grayscaled,45), 3))
cv2.imwrite("45_degree_rotation_interpolated.png", x3_filtering_grayscaled(rotated_grayscale_img(img_as_grayscaled,45), 3))

np.save("q4_90_degree_rotation.npy", rotated_grayscale_img(img_as_grayscaled,90))
cv2.imwrite("90_degree_rotation.png", rotated_grayscale_img(img_as_grayscaled,90))

np.save("q4_135_degree_rotation.npy", rotated_grayscale_img(img_as_grayscaled,135))
cv2.imwrite("135_degree_rotation.png", rotated_grayscale_img(img_as_grayscaled,135))

np.save("q4_135_degree_rotation_interpolated.npy", x3_filtering_grayscaled(rotated_grayscale_img(img_as_grayscaled,135), 3))
cv2.imwrite("135_degree_rotation_interpolated.png", x3_filtering_grayscaled(rotated_grayscale_img(img_as_grayscaled,135), 3))

np.save("q4_180_degree_rotation.npy", rotated_grayscale_img(img_as_grayscaled,180))
cv2.imwrite("180_degree_rotation.png", rotated_grayscale_img(img_as_grayscaled,180))
