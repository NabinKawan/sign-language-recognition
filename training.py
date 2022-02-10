from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os

#load signs, x_test, y_test
testDatasPath=os.path.join("Test_Datas")
signsPath=os.path.join("Signs")
x_test=np.load(os.path.join(testDatasPath,'x_test.npy'))
y_test=np.load(os.path.join(testDatasPath,'y_test.npy'))
signs=np.load(os.path.join(signsPath,'signs.npy'))

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))
model.load_weights('SLR.h5')

res = model.predict(x_test)
print(signs[np.argmax(res[5])])
print(signs[np.argmax(y_test[5])])