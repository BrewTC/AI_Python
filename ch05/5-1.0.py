from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)
txt = LSTM(32)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([att, txt, img], axis=-1) # 用輔助函式串接 3 個張量
out = Dense(28, activation='relu')(out)     # 密集層
out = Dense(1)(out)                         # 輸出層, 可輸出一個預測的銷量值

model = Model([att_in, txt_in, img_in], out)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # 解迴歸問題
wt = model.get_weights()    #←儲存模型的初始權重, 以供稍後重新訓練模型時使用

model.summary()

#用程式產生訓繙資料
import numpy as np

rng = np.random.RandomState(45) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數

    # base的範圍  var的範圍  輸入樣本的 shape
    #         ↓     ↓      ↓
def toy_data(base, var, shape):
    arr = rng.randint(var, size=shape)        # 依 shape 產生 0~(var-1) 的隨機值
    for i in range(shape[0]):                 # 走訪每一個樣本
        arr[i] = arr[i] + rng.randint(base+1) # 將樣本中的每個特徵都加上一個固定的隨機值 (0~base)
    return arr

total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

# 顯示各資料的 shape 及最小、最大值
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))

x_att = x_att.astype('float32') / 100   # 將資料正規化
x_img = x_img.astype('float32') / 255   # 將資料正規化

                  #  以串列依序傳入 3 種樣本資料     分出 20% 做為驗證資料
                  #  -------------------              ↓
history = model.fit([x_att, x_txt, x_img], y, validation_split=0.2,
                    batch_size=128, epochs=1000, verbose=2)

# 訓練方法 2：使用 dict 方式送入資料進行訓練, 鍵為 Input 層的名稱, 值為 Numpy 資料
#model.fit({'att': x_att, 'txt': x_txt, 'img': x_img}, y,
#          validation_split=0.2, batch_size=128, epochs=1000, verbose=2)

import util5 as u   # 匯入自訂模組 (參見 2-0 節的最後單元)

u.plot(history.history, ('mae', 'val_mae'),   #←繪製訓練及驗證的 mae 歷史線圖
       'Training & Validation Mae', ('Epoch','mae'),
       ylim=(0, 300), size=(12, 4))


his = history.history['val_mae']
idx = np.argmin(his)    # 找出最低 val_mae 值的索引
print('最低 val_mae 為第', idx+1, '週期的', his[idx])

    #要轉換的歷史記錄  EMA 係數
    #         ↓      ↓
def to_EMA(points, a=0.3):
  ret = []          # 儲存轉換結果的串列
  EMA = points[0]   # 第 0 個 EMA 值
  for pt in points:
    EMA = pt*a + EMA*(1-a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
    ret.append(EMA)         # 將本期EMA加入串列中
  return ret

his_EMA = to_EMA(history.history['val_mae'])  # 將 val_mae 的值轉成 EMA 值
idx = np.argmin(his_EMA)                      # 找出最低 EMA 值的索引
print('最小 EMA 為第', idx+1, '週期的', his_EMA[idx])


history.history['EMA400'] = his_EMA[400:801]   #取 400~800 的 EMA 資料
history.history['mae400'] = his[400:801]       #取 400~800 的 val_mae 資料
u.plot(history.history, ('EMA400', 'mae400'),  #繪製 EMA 和 val_mae 的比較圖
       'Validation EMA & mae', ('Epoch 400~800','EMA & mae'),
       ylim=(0, 30), size=(12, 4))
             # ↑ y 軸限定只畫出 0~30 的範圍

#####################################################

print(f'用所有的訓練資料重新訓練到第 {idx+1} 週期')
model.set_weights(wt)  #←還原初始權重 (效果等於重建模型, 以便重新訓練)
history = model.fit([x_att, x_txt, x_img], y,     # 用所有的訓練資料重新訓練到第 {idx+1} 週期
          batch_size=128, epochs=idx+1, verbose=2)

print('重新產生 10000 筆測試資料來評估成效')
rng = np.random.RandomState(67) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數
total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

x_att = x_att.astype('float32') / 100   # 將資料正規化
x_img = x_img.astype('float32') / 255   # 將資料正規化

loss, mae = model.evaluate([x_att, x_txt, x_img], y)
print(f'用 10000 筆測試資料評估的結果：mae={round(mae, 3)}')

pred = model.predict([x_att[:3], x_txt[:3], x_img[:3]])
#pred = model.predict({'att':x_att[:3], 'txt':x_txt[:3], 'img':x_img[:3]})
print('預測銷量:', pred.round(1))
print('實際銷量:', y[:3].round(1))