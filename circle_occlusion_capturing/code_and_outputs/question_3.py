# AUTHORS;
# ERTURK OCAK 150200048
# BATURALP INCE 150200738

import cv2
import numpy as np


coins = cv2.imread('Q3.jpg', cv2.IMREAD_COLOR)
coins_with_occlusion = cv2.imread('Q3_2.jpg', cv2.IMREAD_COLOR)

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


def hough_circle_transform(image_RGB, image_after_canny,minDist,param1,param2,minRadius, maxRadius):
    image_copy = image_RGB.copy()
    detect_edges = image_after_canny
    coins_circles = cv2.HoughCircles(detect_edges, cv2.HOUGH_GRADIENT, 1, minDist, param1= param1, param2= param2, minRadius=minRadius, maxRadius=maxRadius)
    coins_circles_converted = np.uint16(np.around(coins_circles))

    for (x,y,r) in coins_circles_converted[0, :]:
        cv2.circle(image_copy, (x,y), r, (255,0,255), 2)
    return image_copy

cv2.imwrite("q3_coins_circled_1.png", hough_circle_transform(coins, change_to_grayscale(coins),80 ,110 ,80 , 20, 60))
cv2.imwrite("q3_coins_circled_2.png", hough_circle_transform(coins, change_to_grayscale(coins),70 ,85, 60, 20, 60))
cv2.imwrite("q3_coins_circled_3.png", hough_circle_transform(coins, change_to_grayscale(coins),60 ,70 ,60 , 20, 60))
cv2.imwrite("q3_coins_circled_4.png", hough_circle_transform(coins, change_to_grayscale(coins),40, 45, 55, 20, 60))
cv2.imwrite("q3_coins_circled_5.png", hough_circle_transform(coins, change_to_grayscale(coins),30 ,35, 40, 20, 60))

cv2.imwrite("q3_coins_with_occlusion_circled_1.png", hough_circle_transform(coins_with_occlusion, change_to_grayscale(coins_with_occlusion),60 ,40 ,25 , 15, 80 ))
cv2.imwrite("q3_coins_with_occlusion_circled_2.png", hough_circle_transform(coins_with_occlusion, change_to_grayscale(coins_with_occlusion),75 ,60 ,40 , 20, 80) )
cv2.imwrite("q3_coins_with_occlusion_circled_3.png", hough_circle_transform(coins_with_occlusion, change_to_grayscale(coins_with_occlusion),90 ,50 ,30 , 30, 80 ))
cv2.imwrite("q3_coins_with_occlusion_circled_4.png", hough_circle_transform(coins_with_occlusion, cv2.Canny(change_to_grayscale(coins_with_occlusion), threshold1=80, threshold2=300),100 ,45 ,35 , 25, 80))
cv2.imwrite("q3_coins_with_occlusion_circled_5.png", hough_circle_transform(coins_with_occlusion, cv2.Canny(change_to_grayscale(coins_with_occlusion), threshold1=80, threshold2=300),100 ,42 ,35 , 10, 80))   
 
