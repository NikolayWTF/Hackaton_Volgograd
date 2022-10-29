import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
import imutils
from PIL import Image
model = tf.keras.models.load_model('best_model.h5')
labels = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]

def find_bbox(image):

    image = imutils.resize(image, width=480, height=480)
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    arrayX = []
    arrayY = []
    arrayW = []
    arrayH = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x -= 2
        y -= 3
        w += 5
        h += 6
        if (h > 10 and w > 10):
            # cv2.imshow('letter', image[y: y + h, x: x + w])
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            i = 0
            v = 1
            while i < len(arrayY) and v:
                if (y < arrayY[i] - 12):
                    arrayX.insert(i, x)
                    arrayY.insert(i, y)
                    arrayW.insert(i, w)
                    arrayH.insert(i, h)
                    v = 0
                else:
                    if (y >= arrayY[i] - 12 and y <= arrayY[i] + 12):
                        j = i
                        vv = 1
                        while j < len(arrayX) and vv:
                            if (y >= arrayY[j] - 12 and y <= arrayY[j] + 12):
                                if (x < arrayX[j] - 5):
                                    arrayX.insert(j, x)
                                    arrayY.insert(j, y)
                                    arrayW.insert(j, w)
                                    arrayH.insert(j, h)
                                    vv = 0
                            else:
                                arrayX.insert(j, x)
                                arrayY.insert(j, y)
                                arrayW.insert(j, w)
                                arrayH.insert(j, h)
                                vv = 0
                            j += 1
                        if (vv == 1):
                            arrayX.append(x)
                            arrayY.append(y)
                            arrayW.append(w)
                            arrayH.append(h)
                        v = 0
                i += 1
            if v == 1:
                arrayX.append(x)
                arrayY.append(y)
                arrayW.append(w)
                arrayH.append(h)
    i = 0
    while i < len(arrayX):
        x = arrayX[i]
        y = arrayY[i]
        w = arrayW[i]
        h = arrayH[i]
        cv2.imwrite("letters/img" + str(i) + ".png", image[y: y + h, x: x + w])
        i += 1

    return len(arrayY)


def img_to_array(img_name):
    img = image.load_img(img_name, target_size=(32,32))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def data_to_tensor(img_name):
    tensors = img_to_array(img_name)
    return np.vstack(tensors)

im1 = cv2.imread('word4.png')
length = find_bbox(im1)
i = 0
answer = ""
#
# tensors = data_to_tensor('mini_letters/img31.png')
# X = tensors.astype("float32") / 255
# predict = model.predict(np.array([X]))
# ind = np.argmax(predict, axis=1)[0]
# print(labels[ind])
# print(predict)

while i < length:
    img = Image.open('letters/img' + str(i) + '.png')
    img = img.resize((32, 32))
    img.save('mini_letters/img' + str(i) + '.png')
    tensors = data_to_tensor('mini_letters/img' + str(i) + '.png')
    X = tensors.astype("float32") / 255
    predict = model.predict(np.array([X]))
    ind = np.argmax(predict, axis=1)[0]
    answer += labels[ind]
    i += 1
print(answer)
