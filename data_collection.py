import cv2
import mediapipe as mp

mpHolistic=mp.solutions.holistic
vid=cv2.VideoCapture(0)

with mpHolistic.Holistic() as holistic:
    while vid.isOpened():
        success,img=vid.read()
        if not success:
            break

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imshow('Collecting data',img)
        # results=ds
        key=cv2.waitKey(1)
        print(key)
        if key == 27 or key == 113: #press esc to close the window
            break

vid.release() 
cv2.destroyAllWindows()