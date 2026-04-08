import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

wCam, hCam = 640, 480
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(
    maxHands=1,
    detectionCon=0.7,
    trackCon=0.7
)

wScr, hScr = pyautogui.size()
pyautogui.FAILSAFE = False

print("Screen:", wScr, hScr)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # Draw frame
    cv2.rectangle(img, (frameR, frameR),
                  (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, _ = detector.findDistance(8, 12, img)

            if length < 40:
                pyautogui.click()
                time.sleep(0.25)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
