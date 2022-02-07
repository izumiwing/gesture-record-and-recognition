import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
pTime = 0
cTime = 0

rFlag = False
rCount = 15
handLmsRecordedX = []
handLmsRecordedY = []
handLmsSample = []

tolerance = 150

while True:
  ret, img = cap.read()
  if ret:
      imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      result = hands.process(imgRGB)

      # print(result.multi_hand_landmarks)
      imgHeight = img.shape[0]
      imgWidth = img.shape[1]

      handTupleX = []
      handTupleY = []

      if result.multi_hand_landmarks:

          for handLms in result.multi_hand_landmarks:
              mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

              for i, lm in enumerate(handLms.landmark):

                  xPos = lm.x * imgWidth
                  yPos = lm.y * imgHeight

                  handTupleX.append(xPos - handLms.landmark[0].x*imgWidth)
                  handTupleY.append(yPos - handLms.landmark[0].y*imgHeight)

      if(handLmsSample != []):
        if(len(handTupleX) == len(handLmsSample[0])):
          lmsA = np.array( [handLmsSample[0],handLmsSample[1]] )

          handTupleXNd = handTupleX
          handTupleYNd = handTupleY

          lmsB = np.array( [handTupleXNd, handTupleYNd] )

          lmsC = abs(lmsA - lmsB)

          deviationPx = 0

          for handLmsG in lmsC:
              for handLmsPx in handLmsG:
                  deviationPx = deviationPx + handLmsPx

          print(lmsC)

          if(deviationPx < tolerance):
              cv2.putText(img, "Deteceted", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


      if(rFlag):

          rCount = rCount - 1
          handLmsRecordedX.append(handTupleX)
          handLmsRecordedY.append(handTupleY)

          if(rCount < 0):
              rFlag = False
              rCount = 15

              handLmsSample = []
              handLmsSample = [np.mean(handLmsRecordedX, axis=0), np.mean(handLmsRecordedY, axis=0)]
              handLmsRecordedX = []
              handLmsRecordedY = []

          cv2.putText(img, "Recording:{}".format(rCount), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

      cTime = time.time()
      fps = 1/(cTime-pTime)
      pTime = cTime
      cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

      cv2.imshow('img', img)

  if cv2.waitKey(1) == ord('q'):
      break
  if cv2.waitKey(1) == ord('r'):
      rFlag = True
  if cv2.waitKey(1) == ord('p'):
      for handLms in result.multi_hand_landmarks:
          print("This handLms: " + handLms.landmark)