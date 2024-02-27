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


first_image_after_canny  = cv2.Canny(change_to_grayscale(first_image_RGB), threshold1=130, threshold2=400)

second_image_after_canny = cv2.Canny(change_to_grayscale(second_image_RGB), threshold1=80, threshold2=280)


def hough_line_transform(image_RGB, image_after_canny, threshold):
    image_copy = image_RGB.copy()
    lines = cv2.HoughLines(image_after_canny, 1, np.pi/180, threshold)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * ( -b))
        y1 = int(y0 + 1000 * (a))


        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_copy,   (x1,y1), (x2,y2), (0,0,255), 2)

    return image_copy


first_image_after_canny  = cv2.Canny(change_to_grayscale(first_image_RGB), threshold1=130, threshold2=400)
second_image_after_canny = cv2.Canny(change_to_grayscale(second_image_RGB), threshold1=80, threshold2=280)

cv2.imwrite('q2_hough_output_first_img_1.jpg', hough_line_transform(first_image_RGB, first_image_after_canny, 150))
cv2.imwrite('q2_hough_output_first_img_2.jpg', hough_line_transform(first_image_RGB, first_image_after_canny, 240))
cv2.imwrite('q2_hough_output_first_img_3.jpg', hough_line_transform(first_image_RGB, first_image_after_canny, 360))

cv2.imwrite('q2_hough_output_second_img_1.jpg', hough_line_transform(second_image_RGB, second_image_after_canny, 150))
cv2.imwrite('q2_hough_output_second_img_2.jpg', hough_line_transform(second_image_RGB, second_image_after_canny, 240))
cv2.imwrite('q2_hough_output_second_img_3.jpg', hough_line_transform(second_image_RGB, second_image_after_canny, 320))
