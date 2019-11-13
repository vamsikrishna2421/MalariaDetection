import cv2,os
#import numpy as np
#import csv
import glob

label = "Uninfected"
#cwd=os.getcwd()
dirList = glob.glob("cell_images/"+label+"/*.png")
file = open("csv/dataset.csv","a")

file.write("Label,area_0,area_1,area_2,area_3,area_4,\n")

for img_path in dirList:
    im = cv2.imread(img_path)

    #apply gaussian blur to smoothen image
    #use mask (5,5). BIgger the mask, more is the blurring.
    im = cv2.GaussianBlur(im,(1,1),2)

    #convert to gray scale image
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(im_gray,127,255,0)
    contours,_ = cv2.findContours(thresh,1,2)

    file.write(label)
    file.write(",")

    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            file.write(str(area))
        except:
            file.write("0")
        
        file.write(",")
    file.write("\n")
    
