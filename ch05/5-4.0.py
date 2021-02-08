from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)s
txt = LSTM(28)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([att, txt, img], axis=-1) # 用輔助函式串接 3 個張量
out = Dense(28, activation='relu')(out)     # 密集層

sell_out = Dense(1, name='sell')(out)     # 迴歸分析的銷量輸出層：輸出預測的銷量
eval_out = Dense(3, activation='softmax', name='eval')(out) # 多元分類的評價輸出層：輸出好評、中評、或負評

model = Model([att_in, txt_in, img_in], [sell_out, eval_out]) # 2 個輸出層

#####↓↓繪製模型圖↓↓#####

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_1.png')  #←繪製模型結構圖

plot_model(model, to_file='model_2.png',  #←繪製包含 shape 但沒有神經層名稱的模型結構圖
           show_shapes=True,
           show_layer_names=False)

