import cv2
import numpy as np

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def threshold(img_gray):
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th
