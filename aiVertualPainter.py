import cv2
import numpy as np
import os
import HandTrackingModule as htm


brushThickness = 12
eraserThickness = 50

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    image = cv2.resize(image, (640, 70))
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)

xp, yp = 0, 0
imgCanvas = np.zeros((360, 640, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    # Detect hands
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]     # Index finger
        x2, y2 = lmList[12][1:]    # Middle finger

        fingers = detector.fingersUp()

        # Selection Mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            if y1 < 70:
                if 50 < x1 < 150:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 170 < x1 < 270:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 290 < x1 < 390:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 410 < x1 < 510:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20),
                          drawColor, cv2.FILLED)

        # Drawing Mode
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 8, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(imgCanvas, (xp, yp), (x1, y1),
                     drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:70, 0:640] = header

    cv2.imshow("AI Virtual Painter", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
