from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense

dnn = Sequential()   #← 建立序列模型
dnn.add(Dense(32, activation='relu', input_dim=48))
dnn.add(Dense(16, activation='relu'))

inp = Input(shape=48)
out = dnn(inp)    #←將序列模型加到目前模型中, 形成巢狀的神經層結構
out = Dense(10)(out)
model = Model(inp, out) # 建立包含巢狀神經層的函數式模型

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_3.png',
           rankdir='LR')  #←由左而右繪製

plot_model(model, to_file='model_4.png',
           rankdir='LR',         #←由左而右繪製
           expand_nested=True)   #←要繪製出巢狀(套疊)神經層的內部結構
