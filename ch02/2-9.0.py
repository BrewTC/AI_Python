from tensorflow.keras.datasets import reuters  # 匯入 reuters 資料集

(a_train, b_train),(a_test, b_test) = reuters.load_data(num_words=10000)

from tensorflow.keras.preprocessing.text import Tokenizer  #←匯入 Tokenizer 類別

tok = Tokenizer(num_words=10000)           #←指定字典的總字數
x_train = tok.sequences_to_matrix(a_train) #←將訓練樣本做 multi-hot 編碼
x_test  = tok.sequences_to_matrix(a_test)  #←將測試樣本做 multi-hot 編碼

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(b_train)  #←將訓練標籤轉為 one-hot 編碼
y_test  = to_categorical(b_test)   #←將測試標籤轉為 one-hot 編碼

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10000))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=512,
                    epochs=20,            #← 訓練 20 個週期
                    verbose=2,     #←顯示精簡訊息 (無進度條)
                    validation_split=0.2) #← 由訓練資料切出 20% 做驗證

import util2 as u
u.plot(history.history, ('acc', 'val_acc'), #←繪製準確率的歷史線圖
       'Training & Validation Acc', ('Epoch','Acc'))

loss, acc = model.evaluate(x_test, y_test, verbose=2)  #←評估訓練成效
print('評估測試資料的準確率 =', acc)

