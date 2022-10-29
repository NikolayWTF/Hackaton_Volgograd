from PIL import Image
from helpers import sliding_window
from functools import reduce
from PIL import ImageChops
import math, operator
import imutils
import keras
import numpy as np
from helpers import sliding_window
import time
import cv2
from matplotlib import pyplot as plt

def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))

labels = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]
image = cv2.imread('pictureSHIA.png')
(winW, winH) = (32, 32)
image = imutils.resize(image, width=80, height=80)
neural_network = keras.models.load_model("best_model.h5")
mask1 = Image.open("picture.png")
mask1 = mask1.convert('1')
minimum = 10000
for (x, y, window) in sliding_window(image, stepSize=2, windowSize=(winW, winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    clone = image.copy()
    cv2.imwrite("tmp_picture.jpg", clone[y : y + winH , x : x + winW])
    im1 = Image.open("tmp_picture.jpg")
    im1 = im1.convert('1')
    res = rmsdiff(im1, mask1)
    if (res < minimum):
        minimum = res
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        print(minimum)
