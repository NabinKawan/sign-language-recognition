import cv2
import mediapipe as mp

mpHolistic=mp.solutions.holistic
mpDraw=mp.solutions.drawing_utils
vid=cv2.VideoCapture(0)

with mpHolistic.Holistic() as holistic:
    while vid.isOpened():
        success,img=vid.read()
        if not success:
            break

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img.flags.writeable=False
        results=holistic.process(img) 
        # img.flags.writeable=True     
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        mpDraw.draw_landmarks(img,results.face_landmarks)
        cv2.imshow('Collecting data',img)
        if cv2.waitKey(1) == 27 : #press esc to close the window
            print(results.face_landmarks)
            break

vid.release() 
cv2.destroyAllWindows()