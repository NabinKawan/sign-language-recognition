import cv2
import mediapipe as mp
import numpy as np
import os
import time

mpHolistic=mp.solutions.holistic
mpDraw=mp.solutions.drawing_utils
vid=cv2.VideoCapture(0)

#path for flatten datas
dataPath = os.path.join('Sign_Data')

key=1

#path for images
imgPath= os.path.join('Sign_Image')

signs=np.array(["Hello","Thank You", "Hungry", "Food", "Hospital", "Washroom"])

#30 video of each sign
numSequences=30

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
    for sequence in range(numSequences):
            try:
                os.makedirs(os.path.join(dataPath,sign,str(sequence)))
                os.makedirs(os.path.join(imgPath,sign,str(sequence)))
            except:
                pass

with mpHolistic.Holistic() as holistic:

    #taking label input 
    while True:
        blankImg = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        cv2.putText(blankImg,"Select label: ", (10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(blankImg,"0: Hello , 1: Thank You", (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(blankImg,"2: Hungry , 3: Food", (10,150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(blankImg,"4: Hospital , 5: Washroom", (10,200),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(blankImg,"'ESC' to escape", (10,250),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Select label',blankImg)
        inpt=cv2.waitKey(0)
        if inpt == 48 or inpt == 49 or inpt == 50 or inpt == 51 or inpt == 52 or inpt == 53:
            choice=signs[inpt-48]
            cv2.destroyWindow('Select label')
            break
        else:
            if inpt==27:
                break
                     
    #checking camera is opened or not and taking data    
    while vid.isOpened() and inpt!=27 :
        for sequence in range(numSequences):
            for frameNum in range(sequenceLength+1):
                #checks for user input to close the windows                
                key=cv2.waitKey(1)                  
                success,img=vid.read()
                
                #checking if data is accessed or not from camera
                if not success:
                    break  
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                # img.flags.writeable=False
                results=holistic.process(img) 
                # img.flags.writeable=True     
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                
                #draws landmarks
                drawLandmarks()
                
                #show feed for collecting datas and delays for 2 sec
                if frameNum == 0: 
                    cv2.putText(img, "Press 'ESC' to escape", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
                    cv2.putText(img, 'Starting collection in 3 sec', (10,60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('Collecting Datas', img)
                    key= cv2.waitKey(3000)
                    
                #starts collecting datas    
                else: 
                    cv2.putText(img, f"Collecting Data for '{choice}' Video Number {sequence}", (15,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
                    cv2.imshow('Collecting Datas', img)
                    keypoints=extractKeypoints()  
                    
                    #saving flatten array to datapath
                    np.save(os.path.join(dataPath,choice,str(sequence),str(frameNum-1)),keypoints)
                    
                    #saving image to image path
                    jpgPath=os.path.join(imgPath,choice,str(sequence),str(frameNum-1))
                    cv2.imwrite(f"{jpgPath}.jpg",img)
                    key=2 #giving default value for key to avoid esc while taking data

                if key == 27 : #press esc to close the window
                    break 

            if key == 27 : #press esc to close the window
                break        
        if key == 27 : #press esc to close the window
            break             
            
#releasing the port            
vid.release() 

#destroying all opened windows using opencv
cv2.destroyAllWindows()            