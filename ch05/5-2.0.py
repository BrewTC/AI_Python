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

total = 10000   # 產生 total 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

y2 = np.ones(total) # 建立評價陣列, 元素值預設為 1 (中評)
att1 = x_att[:, 1]  # 由商品屬性中取出性價比資料
y2[att1>80] = 2   # 性價比大於 80 設為 2 (好評)
y2[att1<20] = 0   # 性價比小於 20 設為 0 (負評)
print('評價 y2 的好評數：', np.sum(y2==2), ', 中評數：', np.sum(y2==1), ', 負評數：', np.sum(y2==0))

print('原來的銷量 y:', y.shape,  ', min =', np.min(y),     ', max =', np.max(y))
y[y2==2] *= 1.5
y[y2==0] *= 0.5
print('調整的銷量 y:', y.shape,  ', min =', np.min(y),     ', max =', np.max(y))

print('total:', total)
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))
print('y2:  :', y2.shape,    ', min =', np.min(y2),    ', max =', np.max(y2))

#建立自訂評量函式
from tensorflow.keras import metrics

def macc(y_true, y_pred):
  return (4000 - metrics.mae(y_true, y_pred)) / 4000

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy'],
              metrics=[['mae', macc], ['acc']],
              loss_weights=[0.01, 100])

#model.compile(optimizer='rmsprop',
#              loss={'sell': 'mse', 'eval': 'categorical_crossentropy'},
#              metrics={'sell': ['mae', macc], 'eval': 'acc'},
#              loss_weights={'sell':0.01, 'eval':100})

wt = model.get_weights()    #←儲存模型的初始權重, 以供稍後重新訓練模型時使用
model.summary()


x_att = x_att.astype('float32') / 100   # 將屬性資料正規化
x_img = x_img.astype('float32') / 255   # 將圖片資料正規化

from tensorflow.keras.utils import to_categorical
y2 = to_categorical(y2)   # 做 one-hot 編碼

                  #  以串列依序傳入 3 種樣本資料     分出 20% 做為驗證資料
                  #  -------------------              ↓
history = model.fit([x_att, x_txt, x_img], [y, y2], validation_split=0.2,
                    batch_size=128, epochs=800, verbose=2)


## 訓練方法 2：使用 dict 方式送入資料進行訓練, 鍵為 Input 層的名稱, 值為 Numpy 資料
#history = model.fit({'att': x_att, 'txt': x_txt, 'img': x_img}, {'sell': y, 'eval': y2},
#                    validation_split=0.2, batch_size=128, epochs=800, verbose=2)

import util5 as u   # 匯入自訂模組 (參見 2-0 節的最後單元)

u.plot(history.history, ('sell_macc', 'val_sell_macc', 'eval_acc', 'val_eval_acc'),   #←繪製訓練及驗證的 mae 歷史線圖
       'Training & Validation Acc', ('Epoch','acc'),
       ylim=(0.6, 1.0), size=(12, 4))

#######################################################
                             # zip() 每次會從各參數中取出一個元素來傳回
his = history.history        # ↓
his_avg = [(a+b)/2 for a, b in zip(his['val_sell_macc'], his['val_eval_acc'])] # 計算 2 種準確率的平均值

idx = np.argmax(his_avg)     # 找出平均準確率最高的索引
print('最高準確率為第', idx+1, '週期的', his_avg[idx],
      ' (銷量：', his['val_sell_macc'][idx], ', 評價：', his['val_eval_acc'][idx], ')')

def to_EMA(points, a=0.3):  # 有關 EMA 的說明請參見前一單元
  ret = []          # 儲存轉換結果的串列
  EMA = points[0]   # 第 0 個 EMA 值
  for pt in points:
    EMA = pt*a + EMA*(1-a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
    ret.append(EMA)         # 將本期EMA加入串列中
  return ret

his_EMA = to_EMA(his_avg)  # 將 his_avg 值轉成 EMA 值
idx = np.argmax(his_EMA)   # 找出最高 EMA 值的索引
print('最高 EMA 為第', idx+1, '週期的', his_EMA[idx],
      ' (銷量：', his['val_sell_macc'][idx],
      ', 評價：', his['val_eval_acc'][idx], ')')

# #####################################################

print(f'用所有的訓練資料重新訓練到第 {idx+1} 週期')
model.set_weights(wt)  #←還原初始權重 (效果等於重建模型, 以便重新訓練)
history = model.fit([x_att, x_txt, x_img], [y, y2],     # 用所有的訓練資料重新訓練到第 {idx+1} 週期
          batch_size=128, epochs=idx+1, verbose=2)

rng = np.random.RandomState(67) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數
total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

y2 = np.ones(total)
att0 = x_att[:, 0]
att1 = x_att[:, 1]

y2[att1>80] = 2
y2[att1<20] = 0
print('好評數：', np.sum(y2==2), ', 中評數：', np.sum(y2==1), ', 負評數：', np.sum(y2==0))

y[y2==2] *= 1.5
y[y2==0] *= 0.5

# 顯示各資料的 shape 及最小、最大值
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))
print('y2:  :', y2.shape,    ', min =', np.min(y2),    ', max =', np.max(y2))

x_att = x_att.astype('float32') / 100
x_img = x_img.astype('float32') / 255
y2 = to_categorical(y2)

res = model.evaluate([x_att, x_txt, x_img], [y, y2], verbose=2)
print(f'用 {total} 筆測試資料評估的結果：{res}')

pred = model.predict([x_att[:3], x_txt[:3], x_img[:3]])
#pred = model.predict({'att':x_att[:3], 'txt':x_txt[:3], 'img':x_img[:3]})

print('預測銷量:', pred[0].round(1))
print('實際銷量:', y[:3].round(1))

print('預測評價:', pred[1].round(1))
print('實際評價:', y2[:3])