# AUTHORS;
# ERTURK OCAK 150200048
# BATURALP INCE 150200738

import cv2
import numpy as np
from matplotlib import image 
from matplotlib import pyplot as plt 


first_image_RGB = cv2.imread("Q1.jpg", cv2.IMREAD_COLOR) 
#first_image_RGB= np.array(cv2.cvtColor(first_image_BGR, cv2.COLOR_BGR2RGB), dtype= np.uint8)


second_image_RGB = cv2.imread("Q1_2.jpg", cv2.IMREAD_COLOR) 
#second_image_RGB= np.array(cv2.cvtColor(second_image_BGR, cv2.COLOR_BGR2RGB), dtype= np.uint8)

def change_to_grayscale(img_as_RGB):
    
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

    return np.array(holder_matrix) 

def x5_gaussian_filtering(grayscaled_img, stride):
    height, width = grayscaled_img.shape
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride)
    gaussian_filtered_img = np.array((zero_padded_grayscaled_img), dtype= np.uint8)


    mask = np.array([[1, 4, 6, 4, 1],
                     [4, 16, 24, 16, 4],
                     [6, 24, 36, 24, 6],
                     [4, 16, 24, 16, 4],
                     [1, 4, 6, 4, 1]])
    mask = mask / 256 

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = (
                        zero_padded_grayscaled_img[i + k - 2, j + k - 2] * mask[0, 0]
                        + zero_padded_grayscaled_img[i + k - 2, j + k - 1] * mask[0, 1]
                        + zero_padded_grayscaled_img[i + k - 2, j + k] * mask[0, 2]
                        + zero_padded_grayscaled_img[i + k - 2, j + k + 1] * mask[0, 3]
                        + zero_padded_grayscaled_img[i + k - 2, j + k + 2] * mask[0, 4]
                        + zero_padded_grayscaled_img[i + k - 1, j + k - 2] * mask[1, 0]
                        + zero_padded_grayscaled_img[i + k - 1, j + k - 1] * mask[1, 1]
                        + zero_padded_grayscaled_img[i + k - 1, j + k] * mask[1, 2]
                        + zero_padded_grayscaled_img[i + k - 1, j + k + 1] * mask[1, 3]
                        + zero_padded_grayscaled_img[i + k - 1, j + k + 2] * mask[1, 4]
                        + zero_padded_grayscaled_img[i + k, j + k - 2] * mask[2, 0]
                        + zero_padded_grayscaled_img[i + k, j + k - 1] * mask[2, 1]
                        + zero_padded_grayscaled_img[i + k, j + k] * mask[2, 2]
                        + zero_padded_grayscaled_img[i + k, j + k + 1] * mask[2, 3]
                        + zero_padded_grayscaled_img[i + k, j + k + 2] * mask[2, 4]
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


def calculate_mean_of_x_of_white_pixels(image_after_canny):
    height,width = image_after_canny.shape
    total_x = 0
    number_of_x = 0

    for x in range(height):
        for y in range(width):
            if(image_after_canny[x][y] == 255):
                total_x += x
                number_of_x = number_of_x + 1

    return total_x/number_of_x


def calculate_mean_of_y_of_white_pixels(image_after_canny):
    height,width = image_after_canny.shape
    total_y = 0
    number_of_y = 0

    for x in range(height):
        for y in range(width):
            if(image_after_canny[x][y] == 255):
                total_y += y
                number_of_y = number_of_y + 1

    return total_y/number_of_y



def calculate_slope_of_best_fitting_line(image_after_canny):
    height = image_after_canny.shape[0] 
    width = image_after_canny.shape[1]

    total_1 = 0
    total_2 = 0
    mean_of_white_x = calculate_mean_of_x_of_white_pixels(image_after_canny)
    mean_of_white_y = calculate_mean_of_y_of_white_pixels(image_after_canny)

    for x in range(height):
        for y in range(width):
            if(image_after_canny[x][y] == 255):
                total_1 += (x - mean_of_white_x) * (y - mean_of_white_y)
                total_2 += (x - mean_of_white_x) * (x - mean_of_white_x)
    
    return total_1/total_2


def draw_line_on_image(image_RGB, slope, intercept):
    height = image_RGB.shape[0]
    width = image_RGB.shape[1]

    for x in range(height):
        y = int(slope * x + intercept)
        if 0 <= y < width:
            image_RGB[x, y] = [0,0,0]

            if(0 <= x+1 < height):
                image_RGB[x+1, y] = [0,0,0]
    
            if(0 <= x-1 < height):
                image_RGB[x-1, y] = [0,0,0]

            if(0 <= x+2 < height):
                image_RGB[x+2, y] = [0,0,0]

            if(0 <= x-2 < height):
                image_RGB[x-2, y] = [0,0,0]

    return image_RGB

def line_detection(image_RGB,outliers):
    if(outliers == False):
        grayscaled_img = change_to_grayscale(image_RGB)
        image_after_canny = cv2.Canny(grayscaled_img, threshold1=130, threshold2=400)
        slope = calculate_slope_of_best_fitting_line(cv2.Canny(change_to_grayscale(image_RGB), threshold1=130, threshold2=400))
        mean_of_white_x = calculate_mean_of_x_of_white_pixels(image_after_canny)
        mean_of_white_y = calculate_mean_of_y_of_white_pixels(image_after_canny)  
        intercept = mean_of_white_y - (slope*mean_of_white_x)
    if(outliers == True):
        grayscaled_img = change_to_grayscale(image_RGB)
        image_after_canny = cv2.Canny(grayscaled_img, threshold1=80, threshold2=280)
        slope = calculate_slope_of_best_fitting_line(cv2.Canny(change_to_grayscale(image_RGB), threshold1=80, threshold2=280))
        mean_of_white_x = calculate_mean_of_x_of_white_pixels(image_after_canny)
        mean_of_white_y = calculate_mean_of_y_of_white_pixels(image_after_canny)
        intercept = mean_of_white_y - (slope*mean_of_white_x)

    draw_line_on_image(image_RGB, slope, intercept)
    return image_RGB


cv2.imwrite("q1_first_image_canny_1.png", cv2.Canny(change_to_grayscale(first_image_RGB), threshold1=120, threshold2=200))
cv2.imwrite("q1_first_image_canny_1_with_gaussian.png", cv2.Canny(x5_gaussian_filtering(change_to_grayscale(first_image_RGB),5), threshold1=120, threshold2=200))
cv2.imwrite("q1_first_image_canny_2.png", cv2.Canny(change_to_grayscale(first_image_RGB), threshold1=120, threshold2=300))
cv2.imwrite("q1_first_image_canny_3.png", cv2.Canny(change_to_grayscale(first_image_RGB), threshold1=130, threshold2=400))

cv2.imwrite("q1_second_image_canny_1.png", cv2.Canny(change_to_grayscale(second_image_RGB), threshold1=80, threshold2=210))
cv2.imwrite("q1_second_image_canny_1_with_gaussian.png", cv2.Canny(x5_gaussian_filtering(change_to_grayscale(second_image_RGB),5), threshold1=80, threshold2=280))
cv2.imwrite("q1_second_image_canny_2.png", cv2.Canny(change_to_grayscale(second_image_RGB), threshold1=90, threshold2=290))
cv2.imwrite("q1_second_image_canny_3.png", cv2.Canny(change_to_grayscale(second_image_RGB), threshold1=83, threshold2=265))

cv2.imwrite("q1_first_image_best_fitting_line.png", line_detection(first_image_RGB, False))
cv2.imwrite("q1_second_image_best_fitting_line.png", line_detection(second_image_RGB, True))
