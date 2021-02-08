from tensorflow.keras import Model                # Model 為函數式 API 的模型類別
from tensorflow.keras.layers import Input, Dense  # 匯入 Input 及 Dense 層類別

A = Input(shape=(784,))                # 將 Input 層的輸出張量 (傳回值) 指定給 A
B = Dense(512, activation='relu')(A)   # 將 A 傳入第一 Dense 層做為輸入張量, 輸出張量指定給 B
C = Dense(10, activation='softmax')(B) # 將 B 傳入第二 Dense 層做為輸入張量, 輸出張量指定給 C

model = Model(A, C)   # 用【最初的輸入張量】和【最後的輸出張量】來建立模型

