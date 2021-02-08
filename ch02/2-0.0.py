from tensorflow.keras.models import Sequential  #← 匯入 Keras 的序列式模型類別
from tensorflow.keras.layers import Dense       #← 匯入 Keras 的密集層類別

model_a = Sequential()                 #← 用 add() 建立序列模型物件
model_a.add(Dense(512, activation='relu', input_dim= 784)) #← 加入第一層
model_a.add(Dense(10, activation='softmax'))               #← 加入第二層

model_b = Sequential([                 #← 建立序列模型物件並加入串列中的層
         Dense(512, activation='relu', input_dim= 784),
         Dense(10, activation='softmax')])
