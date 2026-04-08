import cv2
import numpy as np
import os
import time
import math
import pyautogui
import HandTrackingModule as htm
from cvzone.ClassificationModule import Classifier

wCam, hCam = 640, 480
frameR = 100
smoothening = 7
brushThickness = 12
eraserThickness = 50

wScr, hScr = pyautogui.size()
pyautogui.FAILSAFE = False
displayW = int(wScr * 0.40)
displayH = int(displayW * (hCam / wCam))

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = open("Model/labels.txt").read().splitlines()
imgSize = 300
offset = 20
confirmTime = 1.5

mode = "Mouse"
xp, yp = 0, 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
drawColor = (255, 0, 255)

sentence = ""
currentLetter = ""
confirmedLetter = ""
letterStartTime = 0

modeButtons = {
    "Mouse":   (wCam-150, 80,  wCam-20, 130),
    "Painter": (wCam-150, 150, wCam-20, 200),
    "Letter":  (wCam-150, 220, wCam-20, 270),
}

modeColors = {
    "Mouse": (140,120,255),
    "Painter": (120,255,180),
    "Letter": (120,170,255)
}

colorButtons = [
    ((20, 80),  (70,130), (255,0,255)),
    ((20, 140), (70,190), (255,0,0)),
    ((20, 200), (70,250), (0,255,0)),
    ((20, 260), (70,310), (0,0,0)),
]

def rounded(img, p1, p2, color):
    x1,y1 = p1
    x2,y2 = p2
    cv2.rectangle(img,(x1+12,y1),(x2-12,y2),color,-1)
    cv2.rectangle(img,(x1,y1+12),(x2,y2-12),color,-1)
    for c in [(x1+12,y1+12),(x2-12,y1+12),(x1+12,y2-12),(x2-12,y2-12)]:
        cv2.circle(img,c,12,color,-1)

cv2.namedWindow("GestureFusion", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("GestureFusion", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img,1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    fingers = detector.fingersUp() if lmList else [0]*5

    if lmList:
        x1,y1 = lmList[8][1:]
        for m,(xS,yS,xE,yE) in modeButtons.items():
            if xS<x1<xE and yS<y1<yE and fingers[1] and fingers[2]:
                mode = m
                xp,yp = 0,0
                plocX,plocY = 0,0
                time.sleep(0.25)

    if mode=="Mouse" and lmList:
        if fingers[1] and not fingers[2]:
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            clocX = plocX+(x3-plocX)/smoothening
            clocY = plocY+(y3-plocY)/smoothening
            pyautogui.moveTo(wScr-clocX,clocY)
            plocX,plocY = clocX,clocY

        if fingers[1] and fingers[2]:
            dist,_,_ = detector.findDistance(8,12,img)
            if dist < 40:
                pyautogui.click()
                time.sleep(0.3)

    elif mode=="Painter" and lmList:
        x1,y1 = lmList[8][1:]

        if fingers[1] and fingers[2]:
            for (xA,yA),(xB,yB),col in colorButtons:
                if xA<x1<xB and yA<y1<yB:
                    drawColor = col
                    time.sleep(0.2)

        if fingers[1] and not fingers[2]:
            if xp==0 and yp==0:
                xp,yp=x1,y1
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1
        else:
            xp,yp=0,0

        gray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
        _,inv=cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
        inv=cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)
        img=cv2.bitwise_and(img,inv)
        img=cv2.bitwise_or(img,imgCanvas)

        for (xA,yA),(xB,yB),col in colorButtons:
            cv2.rectangle(img,(xA,yA),(xB,yB),col,-1)
            if drawColor==col:
                cv2.rectangle(img,(xA,yA),(xB,yB),(255,255,255),2)

    elif mode=="Letter" and lmList and bbox:
        x,y,w,h = bbox
        if w>0 and h>0:
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            aspect = h/w
            if aspect>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                imgWhite[:,(imgSize-wCal)//2:(imgSize+wCal)//2] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                imgWhite[(imgSize-hCal)//2:(imgSize+hCal)//2,:] = imgResize

            _, idx = classifier.getPrediction(imgWhite, draw=False)
            detected = labels[idx]

            box_h, box_w = 100, 250
            box = np.zeros((box_h, box_w, 3), np.uint8)  # black box

            cv2.putText(box, f"Current: {detected}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if detected == currentLetter:
                if time.time() - letterStartTime > confirmTime:
                    if detected != confirmedLetter:
                        sentence += detected
                        confirmedLetter = detected
            else:
                currentLetter = detected
                letterStartTime = time.time()

            cv2.putText(box, f"Sentence: {sentence}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Place black box on top-left
            img[10:10+box_h, 10:10+box_w] = box

    panel = img.copy()
    cv2.rectangle(panel,(wCam-170,60),(wCam,290),(30,30,30),-1)
    img = cv2.addWeighted(panel,0.6,img,0.4,0)

    for m,(xS,yS,xE,yE) in modeButtons.items():
        rounded(img,(xS,yS),(xE,yE),
                modeColors[m] if mode==m else (90,90,90))
        cv2.putText(img,m,(xS+30,yS+33),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,0,0) if mode==m else (220,220,220),2)

    imgDisplay = cv2.resize(img,(displayW,displayH),interpolation=cv2.INTER_LINEAR)
    cv2.resizeWindow("GestureFusion", displayW, displayH)
    cv2.imshow("GestureFusion", imgDisplay)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
