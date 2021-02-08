from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten

img_model = Sequential()                             #}
img_model.add(Conv2D(32, (3, 3), activation='relu',  #} 建立序列式模型
              input_shape=(28, 28, 1)))              #} 輸入 shape 為 (28, 28, 1)
img_model.add(Flatten())                             #} 輸出 shape 為 (10,)
img_model.add(Dense(10, activation='softmax'))       #}

#img_model.fit(.....)  # 進行訓練

inp = Input(shape=(28, 28, 1))  # 建立輸入層, 輸入 shape 為 (28, 28, 1)
out = img_model(inp)          # 加入已訓練好的模型 (包含模型中的權重)
out = Dense(10)(out)            # 再連接到 Dense 層
model = Model(inp, out)         # 建立函數式模型
