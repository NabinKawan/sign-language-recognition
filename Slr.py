
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os,cv2
import mediapipe as mp
import time

mpHolistic=mp.solutions.holistic
mpDraw=mp.solutions.drawing_utils
vid=cv2.VideoCapture(0)
sequenceLength=30

#draw landmarks in image
def drawLandmarks():
    mpDraw.draw_landmarks(img,results.face_landmarks,mpHolistic.FACEMESH_CONTOURS,mpDraw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))
    mpDraw.draw_landmarks(img,results.pose_landmarks,mpHolistic.POSE_CONNECTIONS,mpDraw.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mpDraw.draw_landmarks(img,results.left_hand_landmarks,mpHolistic.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(121, 22, 90), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(255, 100, 112), thickness=2, circle_radius=2))
    mpDraw.draw_landmarks(img,results.right_hand_landmarks,mpHolistic.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(250, 80, 10), thickness=1, circle_radius=1),mpDraw.DrawingSpec(color=(200, 50, 250), thickness=2, circle_radius=2))


#extract keypoints from image
def extractKeypoints():
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten()if results.face_landmarks else np.zeros(468*3)
    leftHand=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()if results.left_hand_landmarks else np.zeros(21*3)
    rightHand=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()if results.right_hand_landmarks else np.zeros(21*3)
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([face,leftHand,rightHand,pose])

#load signs, x_test, y_test
# testDatasPath=os.path.join("Test_Datas")
signsPath=os.path.join("Signs")
# x_test=np.load(os.path.join(testDatasPath,'x_test.npy'))
# y_test=np.load(os.path.join(testDatasPath,'y_test.npy'))
signs=np.load(os.path.join(signsPath,'signs.npy'))



#load model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))  # 128
model.add(LSTM(128, return_sequences=True, activation='relu'))  # 64
model.add(LSTM(64, return_sequences=False, activation='relu'))  # 64
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))
model.load_weights('SLR.h5')
print(model.summary())
# real time testing
sequence=[]
sentence=[]
predictions = []
threshold=0.5
prediction_duration=[]
start_time=0.0
end_time=0.0
start_timer=True

dummy_arr=np.zeros([10,1662],dtype=float)

dummy_lst=[]

for e in dummy_arr:
    dummy_lst.append(list(e))

# ['Hello' 'Thank You' 'Hungry' 'Food' 'Hospital' 'Washroom']
def get_threshold(index):
    switcher = {
        0: 1.0, # for Hello
        1: 0.5,  # for Thank You
        2: 0.5,  # for Hungry
        3: 0.02,  # for Food
        4: 0.3,  # for Hospital
        5: 0.3   # for Washroom
    } 
    return switcher.get(index)


with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                     
    #checking camera is opened or not and taking data    
    while vid.isOpened():
        if start_timer:
            start_time=time.time()
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

        #starts collecting datas    
        keypoints=extractKeypoints() 

        #appending to sequence list 
        sequence.append(keypoints)
        print(len(sequence))
        sequence = sequence[-30:]
        if len(sequence)==30:
            #prediction   
            # print(np.array(sequence).shape)
            # print(np.expand_dims(sequence, axis=0).shape)
            # temp_sequence=sequence
            # temp_sequence.append(dummy_lst)
            res=model.predict(np.expand_dims(sequence, axis=0))[0]
            print(res)
            predictions.append(np.argmax(res))
            
            #res=[1.0 if x>1 else x for x in res ] 
            # print(res)
            print(signs[np.argmax(res)])
            # threshold_val=get_threshold(np.argmax(res))
            if np.unique(predictions[-20:])[0]==np.argmax(res): 
                
                print(predictions[-20:])
                if res[np.argmax(res)] > threshold:
                     

                    if len(sentence) > 0: 
                        if signs[np.argmax(res)] != sentence[-1]:
                            sentence.append(signs[np.argmax(res)])
                            start_timer=True
                            end_time=time.time()
                            prediction_duration.append([signs[predictions[-1]],round(end_time-start_time,2)])

                        else:
                            start_timer=False    
                            
                    else:
                        end_time=time.time()
                        prediction_duration.append([signs[predictions[-1]],round(end_time-start_time,2)])
                        sentence.append(signs[np.argmax(res)])
                        start_timer=True
            else:
                start_timer=False
        if len(sentence) > 5: 
            sentence =[]
            # cv2.putText('Collecting Datas',"Select label: ", (10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA) 
            # cv2.imshow('Collecting Datas', img)
        
        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)     
        cv2.putText(img,' '.join(sentence), (10,25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) 
        cv2.imshow('SLR',img)
    
            
        if key == 27 : #press esc to close the window
            break   
            
#releasing the port            
vid.release() 

#destroying all opened windows using opencv
cv2.destroyAllWindows()

print(prediction_duration)