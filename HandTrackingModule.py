import cv2
import mediapipe as mp
import math


class handDetector:
    def __init__(self,
                 mode=False,
                 maxHands=2,
                 detectionCon=0.7,
                 trackCon=0.7):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []

        if not self.results.multi_hand_landmarks:
            return self.lmList, bbox

        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList, bbox

        myHand = self.results.multi_hand_landmarks[handNo]
        h, w, _ = img.shape
        xList, yList = [], []

        for id, lm in enumerate(myHand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.lmList.append([id, cx, cy])
            xList.append(cx)
            yList.append(cy)

            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if xList and yList:
            bbox = min(xList), min(yList), max(xList), max(yList)
            if draw:
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0), 2
                )

        return self.lmList, bbox

    def fingersUp(self):
        fingers = [0, 0, 0, 0, 0]

        if len(self.lmList) != 21:
            return fingers

        # Thumb
        fingers[0] = 1 if self.lmList[4][1] > self.lmList[3][1] else 0

        # Other fingers
        fingers[1] = 1 if self.lmList[8][2] < self.lmList[6][2] else 0
        fingers[2] = 1 if self.lmList[12][2] < self.lmList[10][2] else 0
        fingers[3] = 1 if self.lmList[16][2] < self.lmList[14][2] else 0
        fingers[4] = 1 if self.lmList[20][2] < self.lmList[18][2] else 0

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        if len(self.lmList) != 21:
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
