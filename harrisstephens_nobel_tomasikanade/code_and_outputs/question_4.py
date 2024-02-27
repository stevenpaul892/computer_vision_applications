# AUTHORS;
# ERTURK OCAK 150200048
# BATURALP INCE 150200738

import cv2
import numpy as np


sharp_image_RGB = cv2.imread('Q4.png', cv2.IMREAD_COLOR)
#sharp_image_RGB= np.array(cv2.cvtColor(sharp_image_BGR, cv2.COLOR_BGR2RGB), dtype= np.uint8)

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

    return np.array(holder_matrix, dtype = np.float64)  

def x_derivative_of_img_using_sobel(img_as_RGB,stride):
    grayscaled_img = change_to_grayscale(img_as_RGB)
    height, width = grayscaled_img.shape
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride)
    x_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64)
    mask = np.array([[1,2,1],
                     [0,0,0],
                     [-1,-2,-1]])

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            x_derivative_of_img[i+k][j+k] = weighted_mean_around
    
    return x_derivative_of_img

def y_derivative_of_img_using_sobel(img_as_RGB,stride):
    grayscaled_img = change_to_grayscale(img_as_RGB)
    height, width = grayscaled_img.shape
    zero_padded_grayscaled_img = zero_padding_grayscaled(grayscaled_img,stride)
    y_derivative_of_img = np.array((zero_padded_grayscaled_img), dtype= np.float64)
    mask = np.array([[1,0,-1],
                     [2,0,-2],
                     [1,0,-1]])

    k = int((stride-1)/2)
    for i in range (height):
        for j in range (width):
            weighted_mean_around = zero_padded_grayscaled_img[i+k-1, j+k-1]*mask[0, 0] + zero_padded_grayscaled_img[i+k-1, j+k]*mask[0, 1] + zero_padded_grayscaled_img[i+k-1, j+k + 1]*mask[0, 2] + zero_padded_grayscaled_img[i+k, j+k-1]*mask[1, 0] + zero_padded_grayscaled_img[i+k, j+k]*mask[1, 1] + zero_padded_grayscaled_img[i+k, j+k + 1]*mask[1, 2] + zero_padded_grayscaled_img[i+k + 1, j+k-1]*mask[2, 0] + zero_padded_grayscaled_img[i+k + 1, j+k]*mask[2, 1] +  zero_padded_grayscaled_img[i+k + 1, j+k + 1]*mask[2, 2]
            y_derivative_of_img[i+k][j+k] = weighted_mean_around
    
    return y_derivative_of_img


def HarrisStephens(M):

    determinant_of_M = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    trace_of_M = M[0][0] + M[1][1]
    alpha_value = 0.035

    R = determinant_of_M - alpha_value * (trace_of_M ** 2)
    return R


def TomasiKanade(M):
    # Tomasi-Kanade cornerness measure
    lambda1 = (M[0, 0] + M[1, 1] - ((M[0, 0] - M[1, 1])**2 + 4 * M[0, 1] * M[1, 0])**0.5) / 2
    lambda2 = (M[0, 0] + M[1, 1] + ((M[0, 0] - M[1, 1])**2 + 4 * M[0, 1] * M[1, 0])**0.5) / 2
    return min(lambda1, lambda2)

def Nobel(M, epsilon=1e-6):
    # Nobel cornerness measure
    det_M = np.linalg.det(M)
    trace_M = np.trace(M)
    return det_M / (trace_M + epsilon)

def calculate_eigenvalue(M):
    trace_M = M[0, 0] + M[1, 1]
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    lambda1 = (trace_M + np.sqrt(trace_M**2 - 4 * det_M)) / 2
    lambda2 = (trace_M - np.sqrt(trace_M**2 - 4 * det_M)) / 2

    return lambda1,lambda2


Ix = x_derivative_of_img_using_sobel(sharp_image_RGB,3)
Iy = y_derivative_of_img_using_sobel(sharp_image_RGB,3)

def corner_detection(image_RGB, corner_detection_method,threshold, direction_of_changes):
    
    image_copy = image_RGB.copy()
    grayscaled_img = change_to_grayscale(image_copy)
    height,width = grayscaled_img.shape
    M_matrix = np.zeros((2, 2), dtype=np.float64)
    direction_counts = {'Horizontal': 0, 'Vertical': 0, 'Diagonal': 0}

    for x in range(1, height - 1):
        for y in range(1, width - 1):
            patch_Ix = Ix[x-1:x+2, y-1:y+2]
            patch_Iy = Iy[x-1:x+2, y-1:y+2]

            M_matrix[0, 0] = np.sum(patch_Ix * patch_Ix)
            M_matrix[0, 1] = np.sum(patch_Ix * patch_Iy)
            M_matrix[1, 0] = np.sum(patch_Ix * patch_Iy)
            M_matrix[1, 1] = np.sum(patch_Iy * patch_Iy)

            HarrisStephens_R = HarrisStephens(M_matrix)
            TomasiKanade_R = TomasiKanade(M_matrix)
            Nobel_R = Nobel(M_matrix)
            if(corner_detection_method == "HarrisStephens"):
                if(HarrisStephens_R > threshold):
                    image_copy[x][y] = [0, 0 , 255]
            
            if(corner_detection_method == "TomasiKanade"):
                if(direction_of_changes == True):
                    eigenvalue1,eigenvalue2 = calculate_eigenvalue(M_matrix)
                    min_direction = np.arctan2(M_matrix[1, 0] - eigenvalue1, M_matrix[0, 0] - eigenvalue1)
                    max_direction = np.arctan2(M_matrix[1, 0] - eigenvalue2, M_matrix[0, 0] - eigenvalue2)

                    if max_direction > threshold:
                        angle_diff = abs(max_direction - min_direction) % np.pi 
                        if np.isclose(angle_diff, 0) or np.isclose(angle_diff, np.pi):
                            direction_counts['Vertical'] += 1
                        elif np.isclose(angle_diff, np.pi / 2):
                            direction_counts['Horizontal'] += 1
                        else:
                            direction_counts['Diagonal'] += 1

                        cv2.line(image_copy, (y, x), (int(y + np.cos(max_direction) * 10), int(x + np.sin(max_direction) * 10)), (0, 255, 0), 2)

                    if min_direction > threshold:
                        angle_diff = abs(min_direction - max_direction) % np.pi 
                        if np.isclose(angle_diff, 0) or np.isclose(angle_diff, np.pi):
                            direction_counts['Vertical'] += 1
                        elif np.isclose(angle_diff, np.pi / 2):
                            direction_counts['Horizontal'] += 1
                        else:
                            direction_counts['Diagonal'] += 1

                        cv2.line(image_copy, (y, x), (int(y + np.cos(min_direction) * 10), int(x + np.sin(min_direction) * 10)), (0, 0, 255), 2)

                else:
                    if(TomasiKanade_R >threshold):
                        image_copy[x][y] = [0, 255 , 0]
            
            if(corner_detection_method == "Nobel"):
                if(Nobel_R > threshold):
                    image_copy[x][y] = [255, 0 , 0]
    
    if(corner_detection_method == "TomasiKanade" and direction_of_changes == True ):
        predominant_direction = max(direction_counts, key=direction_counts.get)
        text = f"Predominant Change Direction: {predominant_direction}"
        cv2.putText(image_copy, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 255), 1)

    return image_copy

cv2.imwrite('q4_HarrisStephens_1.jpg', corner_detection(sharp_image_RGB,"HarrisStephens",1, False))
cv2.imwrite('q4_HarrisStephens_10.jpg', corner_detection(sharp_image_RGB,"HarrisStephens",10, False))
cv2.imwrite('q4_HarrisStephens_1000.jpg', corner_detection(sharp_image_RGB,"HarrisStephens",1000, False))
cv2.imwrite('q4_HarrisStephens_10000.jpg', corner_detection(sharp_image_RGB,"HarrisStephens",10000, False))


cv2.imwrite('q4_TomasiKanade_40.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 40, False))
cv2.imwrite('q4_TomasiKanade_2500.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 2500, False))
cv2.imwrite('q4_TomasiKanade_10000.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 10000, False))
cv2.imwrite('q4_TomasiKanade_100000.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 100000, False))

cv2.imwrite('q4_TomasiKanade_40_analysis.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 40, True))
cv2.imwrite('q4_TomasiKanade_2500_analysis.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 2500, True))
cv2.imwrite('q4_TomasiKanade_10000_analysis.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 10000, True))
cv2.imwrite('q4_TomasiKanade_100000_analsis.jpg', corner_detection(sharp_image_RGB,"TomasiKanade", 100000, True))

cv2.imwrite('q4_Nobel_40.jpg', corner_detection(sharp_image_RGB,"Nobel",40, False))
cv2.imwrite('q4_Nobel_2500.jpg', corner_detection(sharp_image_RGB,"Nobel",2500, False))
cv2.imwrite('q4_Nobel_15000.jpg', corner_detection(sharp_image_RGB,"Nobel",15000, False))
cv2.imwrite('q4_Nobel_80000.jpg', corner_detection(sharp_image_RGB,"Nobel",80000, False))




