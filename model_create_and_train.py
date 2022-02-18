import cv2
import mediapipe as mp
import numpy as np
import os


#path for flatten datas
dataPath = os.path.join('Sign_Data')

#path for images
imgPath= os.path.join('Sign_Image')

signs=np.array(["Accident","Ambulance","Clap","Fight","Help","Hospital","Hungry","Ill","Medicine","Washroom",])

#30 video of each sign
numSequences=90

#length of each video
sequenceLength=30

for sign in signs:
    for sequence in range(numSequences):
            try:
                os.makedirs(os.path.join(dataPath,sign,str(sequence)))
                os.makedirs(os.path.join(imgPath,sign,str(sequence)))
            except:
                pass
# if not os.path.exists("Sign_Data"):
#     os.mkdir("Sign_Data")
# else:
#     print("already exist")


from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(signs)}


sequences, labels = [], []
for sign in signs:
    for sequence in range(numSequences):
        window = []
        for frameNum in range(sequenceLength):
            res = np.load(os.path.join(dataPath, sign, str(sequence), "{}.npy".format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[sign])

X = np.array(sequences)
y = to_categorical(labels).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


testDatasPath=os.path.join("Test_Datas")
signsPath=os.path.join("Signs")
try:
    os.makedirs(os.path.join(testDatasPath))
    os.makedirs(os.path.join(signsPath))
except:
    pass
np.save(os.path.join(testDatasPath,'x_test.npy'),X_test)
np.save(os.path.join(testDatasPath,'y_test.npy'),y_test)
np.save(os.path.join(signsPath,'signs.npy'),signs)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard

def createModel():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(signs.shape[0], activation='softmax'))
    return model


model=createModel()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model.fit(X_train, y_train, epochs=1500, callbacks=[tb_callback])