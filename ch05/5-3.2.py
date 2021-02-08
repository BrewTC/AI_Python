from keras.models import Model
from keras.layers import Input, LSTM, Dense, concatenate

                     #↓不定長度的句子,以 256 字的字典編碼
a_in = Input(shape=(None, 256))  # 分支 a 的輸入
b_in = Input(shape=(None, 256))  # 分支 b 的輸入

shared_lstm = LSTM(64)  # 建立共用的 LSTM 層

a = shared_lstm(a_in)   # 建立分支 a, 輸出張量 shape 為 (批次量, 64)
b = shared_lstm(b_in)   # 建立分支 a, 輸出張量 shape 為 (批次量, 64)

c = concatenate([a, b]) # 將二分支的輸出串接起來
out = Dense(1, activation='sigmoid')(c)  # 進行 2 元分類 (是否意義相同？)

model = Model(inputs=[a_in, b_in], outputs=out)  # 建立模型

model.compile(optimizer='rmsprop', loss='binary_crossentropy',  # 編譯模型
              metrics=['acc'])

model.summary()

