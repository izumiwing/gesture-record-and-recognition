# Gesture Record & Recognition

Simply using Nump,cv2,MediaPipe to make this come true.

According to Mediapipe wikipage,we have know mp will acquire a Hand Landmark while finding a image containg a hand.![fixed_point](https://user-images.githubusercontent.com/19906645/152786752-602c07a3-0d8d-4bf7-b78d-c9bad06cbad1.png)
so we can just pick any point and minus the row X/Y to calculate the other points position.
```
handTupleX.append(xPos - handLms.landmark[0].x*imgWidth)
handTupleY.append(yPos - handLms.landmark[0].y*imgHeight)
```
Press 'R' to record a simple gesture.

I have started timer like if loop to capture the image repeatedly from Cam then we use Numpy to calculate the average.
```
handLmsSample = [np.mean(handLmsRecordedX, axis=0), np.mean(handLmsRecordedY, axis=0)]
```

When you recorded a gesture,script will compare your real-time capture with the recorded sample:
```
lmsA = np.array( [handLmsSample[0],handLmsSample[1]] )

handTupleXNd = handTupleX
handTupleYNd = handTupleY

lmsB = np.array( [handTupleXNd, handTupleYNd] )

lmsC = abs(lmsA - lmsB)

deviationPx = 0

for handLmsG in lmsC:
    for handLmsPx in handLmsG:
        deviationPx = deviationPx + handLmsPx

```
Basically,the gesture detecteing is through the every single finger's X and Y after minus the Point X. if you want to make the generally,you can adjust the tolerance variable you like.
```
tolerance = 50 # strict
tolerance = 150 # loose
# No matter what number i have given,you can just adjust whatever you like.
```
