import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

offset = 20
imgSize = 300

currentLetter = ""
confirmedLetter = ""
sentence = ""

startTime = 0
confirmationTime = 1.5  # seconds (change to 2.0 if you want slower)

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()

    hands, img = detector.findHands(img, draw=False)

    detectedLetter = ""

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if 0 <= index < 26:
                detectedLetter = labels[index]

            if detectedLetter == currentLetter:
                if time.time() - startTime >= confirmationTime:
                    if detectedLetter != confirmedLetter:
                        sentence += detectedLetter
                        confirmedLetter = detectedLetter
            else:
                currentLetter = detectedLetter
                startTime = time.time()

            # Draw bounding box
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                (255, 0, 255),
                3
            )

            # Show current letter
            cv2.putText(
                imgOutput,
                f"Detected: {detectedLetter}",
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("ImageWhite", imgWhite)

    cv2.rectangle(imgOutput, (0, 0), (imgOutput.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(
        imgOutput,
        f"Sentence: {sentence}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    # Press SPACE to add space
    if key == 32:
        sentence += " "
        confirmedLetter = ""

    # Press BACKSPACE to delete
    if key == 8:
        sentence = sentence[:-1]

    # ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
