import cv2
import numpy as np
import os
import time

def whitemask(hsv, lowlimit):
    lower_white = np.array([0, 0, lowlimit])                  #Hsv lower limit
    upper_white = np.array([225, 25, 255])               #Hsv upper limit
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

def bwtorgb(newmask):
    newmaskrgb = np.array([newmask, newmask, newmask])
    newmaskrgb = newmaskrgb.swapaxes(0, 1)
    newmaskrgb = newmaskrgb.swapaxes(1,2)
    return newmaskrgb

def crop(newmask, img2):
    img3 = np.zeros(img2.shape)
    for i in range(1, img2.shape[0] - 1):
        for j in range(1, img2.shape[1] - 1):
            sum = 0
            for q in range(2):
                for w in range(2):
                    sum = sum + newmask[(i + 1 - q), (j + 1 - w)]
            if sum > 255:
                img3[i, j] = img2[i, j]
    return img3

def genedispersion(img, dispersion, size):
    newimg = np.zeros(img.shape)
    for i in range(img.shape[0] - 5):
        for j in range(img.shape[1] - 5):
            if img[i, j] == 255:
                sum = 0
                for q in range(10):
                    for w in range(10):
                        sum = sum + img[(i + 5 - q), (j + 5 - w)]
                if sum > dispersion * 255:
                    for q in range(size):
                        for w in range(size):
                            newimg[(i + int(size / 2) - q), (j + int(size / 2) - w)] = 255
    return newimg

path2 = 'data/all_data/train/images/'
writepath3 = 'data/all_data/train/images_rgb/'
files = os.listdir(path2)
dispersion = 15
size = 5
groundhsv = cv2.imread('color_square.png')
for file in files:
    count = count + 1
    start = time.time()
    img2 = cv2.imread(path2 + file)
    img3 = whitemask(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV), 120)
    img4 = genedispersion(img3, dispersion, size)
    groundhsv = cv2.resize(groundhsv, (img2.shape[1], img2.shape[0]))
    img5 = crop(img4, (img2 + groundhsv))
    cv2.imwrite(writepath3 + file, img5)
