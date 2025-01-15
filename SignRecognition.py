import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["alef", "ba", "ta", "tha", "jem", "h7a", "kha", "dal", "thal", "ra", "zay", "sen", "shen", "sad", "dhad",
          "dta", "dtha", "aen", "gen", "fa", "qaf", "kaf", "lam", "mem", "nun", "ha", "waw", "ya", "space"]
labels1 = ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض",
           "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", " "]
word = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    reshaped_word = arabic_reshaper.reshape(word)
    bidi_text = get_display(reshaped_word)
    fontpath = "arial.ttf"  # <== https://www.freefontspro.com/14454/arial.ttf
    font = ImageFont.truetype(fontpath, 32)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction[index]*100)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction[index]*100)


        #cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      #(x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 255), 2)
        #cv2.rectangle(imgOutput, (x - offset, y - offset),
                      #(x + w + offset, y + h + offset), (255, 0, 255), 4)

    img_pil = Image.fromarray(imgOutput)
    draw = ImageDraw.Draw(img_pil)
    draw.text((50, 80), bidi_text, font=font)
    img = np.array(img_pil)

    # cv2.imshow("Image", imgOutput)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    # Add letter
    if key == ord('a'):
        letter = labels1[index]
        word = word + letter
        print(word)

    # Backspace
    elif key == ord('b'):
        word = word[:-1]
        print(word)

    # Clear the sentence
    elif key == ord('c'):
        word = "                         "
        word = ""
        print(word)
