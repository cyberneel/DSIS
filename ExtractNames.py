# import dependencies
import easyocr
import cv2
import os
import numpy as np

# the ocr object that runs detection
ocr = easyocr.Reader(['en'], gpu=True)

current_dir = os.path.dirname(__file__)

def preprocessImageEdges(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def ExtractNamesText(imgPath=None):
    # return if no path
    if imgPath == None:
        return
    
    # have the paths for the images
    imgPath = os.path.join(current_dir, imgPath)

    # read the image
    image = cv2.imread(imgPath)

    # run the OCR
    baseResult = ocr.readtext(imgPath, mag_ratio=8.0, adjust_contrast=2)

    print(baseResult)