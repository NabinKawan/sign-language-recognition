import cv2
import mediapipe as mp
import numpy as np
import os

mpHolistic=mp.solutions.holistic
mpDraw=mp.solutions.drawing_utils
vid=cv2.VideoCapture(0)
signs=np.array(["Hello","Thank You"])

#30 video of each sign
noSequences=30

#length of each video
sequenceLength=30

#draw landmarks in image
def drawLandmarks():
    mpDraw.draw_landmarks(img,results.face_landmarks,mpHolistic.FACEMESH_CONTOURS,mpDraw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))
    mpDraw.draw_landmarks(img,results.pose_landmarks,mpHolistic.POSE_CONNECTIONS,mpDraw.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mpDraw.draw_landmarks(img,results.left_hand_landmarks,mpHolistic.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(121, 22, 90), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(255, 100, 112), thickness=2, circle_radius=2))
    mpDraw.draw_landmarks(img,results.right_hand_landmarks,mpHolistic.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(250, 80, 10), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(200, 50, 250), thickness=2, circle_radius=2))

# extract keypoints of the features in image
def extractKeypoints():
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten()if results.face_landmarks else np.zeros(468*3)
    leftHand=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()if results.left_hand_landmarks else np.zeros(21*3)
    rightHand=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()if results.right_hand_landmarks else np.zeros(21*3)
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([face,leftHand,rightHand,pose])

# making folders if not exist
for sign in signs:
    for sequence in range(noSequences):
            try:
                os.makedirs(os.path.join("Sign_Data",sign,str(sequence)))
            except:
                pass
# if not os.path.exists("Sign_Data"):
#     os.mkdir("Sign_Data")
# else:
#     print("already exist")

with mpHolistic.Holistic() as holistic:
    #checking is camera is opened or not
    while vid.isOpened():
        success,img=vid.read()
        #checking if data is accessed or not from camera
        if not success:
            break
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img.flags.writeable=False
        results=holistic.process(img) 
        # img.flags.writeable=True     
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        drawLandmarks()
        #shows image in frame
        cv2.imshow('Collecting data',img)
        if cv2.waitKey(1) == 27 : #press esc to close the window
            print(len(results.face_landmarks.landmark))
            break
            
#releasing the port            
vid.release() 

#destroying all opened windows using opencv
cv2.destroyAllWindows() 