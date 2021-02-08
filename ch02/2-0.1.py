from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation  #← 匯入啟動函數層類別

model = Sequential([
        Dense(512, input_dim= 784), #← 第一密集層不指定 (省略) 啟動函數
        Activation('relu'),         #← 接著加入 relu 啟動函數層
        Dense(10),                  #← 第二密集層同樣不指定啟動函數
        Activation('softmax')])     #← 接著加入 softmax 啟動函數層
model.summary()
