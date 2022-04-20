import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

images = []

for filename in glob.glob('C:/Git/codetest_datascience/data/*.jpg'):

    input_image = cv2.imread(filename)  

    output_image = np.copy(input_image)

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    ksize = (30, 30)
    blur_image = cv2.blur(gray_image, ksize)
    edges_image = cv2.Canny(blur_image,10,20)

    small_circles = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 100, param1=7, param2=90, minRadius=50, maxRadius=150) 
    medium_circles = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 150, param1=7, param2=85, minRadius=150, maxRadius=250)  
    big_circles = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 250, param1=7, param2=85, minRadius=250, maxRadius=350)     
    
    if small_circles is not None:
        small_circles = small_circles[0, :].astype("int")
        for (x, y, r) in small_circles:
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 4)
            
    if medium_circles is not None:
        medium_circles = medium_circles[0, :].astype("int")
        for (x, y, r) in medium_circles:
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 4)
            
    if big_circles is not None:
        big_circles = big_circles[0, :].astype("int")
        for (x, y, r) in big_circles:
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 4)

    images.append(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    images.append(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
 
idx = 0
for image in images:    
    plt.figure(figsize=(40,20))
    plt.imshow(image)
    plt.savefig('C:/Git/codetest_datascience/output/image_'+str(idx)+'.png')
    plt.show()
    idx += 1

