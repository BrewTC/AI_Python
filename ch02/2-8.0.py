from tensorflow.keras.datasets import imdb    #← 從 keras.datasets 套件中匯入 imdb
(a_train, b_train),(a_test, b_test)= imdb.load_data(num_words=10000) # 載入 IMDB

from tensorflow.keras.preprocessing.text import Tokenizer

tok = Tokenizer(num_words=10000)           #←指定字典的總字數
x_train = tok.sequences_to_matrix(a_train) #←將訓練樣本做 k-hot 編碼
x_test  = tok.sequences_to_matrix(a_test)  #←將測試樣本做 k-hot 編碼

y_train = b_train.astype('float32')   #←將訓練標籤轉為浮點向量
y_test  = b_test.astype('float32')    #←將測試標籤轉為浮點向量

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()                       #←建立模型物件
model.add(Dense(16, activation='relu', input_dim=10000))  #←輸入層
model.add(Dense(16, activation='relu'))    #←隱藏層
model.add(Dense(1, activation='sigmoid'))  #←輸出層

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=512,  #←每批次 512 筆樣本
                    epochs=10,       #←共訓練 10 週期
                    verbose = 2,     #←顯示精簡訊息 (無進度條)
                    validation_split=0.2)
                             #↑由訓練資料後面切出 20% 做為驗證用

import util2 as u

u.plot(history.history,
       ('loss', 'val_loss'),          #←歷史資料中的 key
       'Training & Validation Loss',  #←線圖的標題
       ('Epoch','Loss'))              #←x,y 軸的名稱
u.plot(history.history,
       ('acc', 'val_acc'),            #←歷史資料中的 key
       'Training & Validation Acc',   #←線圖的標題
       ('Epoch','Acc'))               #←x,y 軸的名稱
