import pandas as pd
import util2 as u     # 匯入自訂的工具模組

df = pd.read_csv("Admission_Predict_Ver1.1.csv",sep = ",")

import numpy as np

np.random.seed(6)       # 設定亂數的種子, 以便每次執行產生的亂數序列都相同
ds = df.values          # 取出 DataFrame 中的資料 (不含標題欄)
np.random.shuffle(ds)   # 將所有資料洗牌 (隨機重排)

x = ds[:, 1:8]   # 取出所有資料列的第 1 到 7 欄資料做為樣本資料
y = ds[:, 8]     # 取出所有資料列的第 8 欄資料做為標籤資料

x_train = x[:400]  #} 前 400 筆 (0~399) 做為訓練用
y_train = y[:400]  #}
x_test  = x[400:]  #} 後 100 筆 (400~499) 做為測試用
y_test  = y[400:]  #}

###############################

mean = x_train.mean(axis=0)  #←沿著第 0 軸 (樣本數軸) 對每個特徵做平均, 因此 mean.shape 為 (13,)
std = x_train.std(axis=0)    #←沿著第 0 軸 (樣本數軸) 對每個特徵算標準差, std.shape 同樣為 (13,)

x_train -= mean  #← 將訓練樣本減平均值
x_train /= std   #← 再除以標準差

x_test  -= mean  #← 將測試樣本減平均值
x_test  /= std   #← 再除以標準差

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=7))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))           #←不加任何啟動函數
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

wt = model.get_weights()    #←儲存模型的初始權重
ksize = len(x_train) // 4   #←計算每折的資料筆數
all_his_mae = []   #←建立變數來儲存每次訓練的 mae 歷史資料
all_his_val = []   #←建立變數來儲存每次訓練的 val_mae 歷史資料

#    print('開始 k 摺驗證', end='')
for i in range(4):
    print(f'第 {i} 摺訓練與驗證', end='')
    fr, to = i*ksize, (i+1)*ksize  #←計算驗證資料的起點 fr 和終點 to (不含)
    x_val = x_train[fr: to]        #←取出驗證樣本
    x_trn = np.concatenate([x_train[ :fr],  #←取出訓練樣本：取驗證資料以外的部份
                            x_train[to: ]], axis=0)
    y_val = y_train[fr: to]        #←取出驗證標籤
    y_trn = np.concatenate([y_train[ :fr],  #←取出訓練標籤：取驗證資料以外的部份
                            y_train[to: ]], axis=0)

    model.set_weights(wt)  #←還原初始權重 (效果等於重建模型, 以便重新訓練)
    history =  model.fit(x_trn, y_trn,
                         validation_data=(x_val, y_val),  #←指定驗證資料
                         epochs=200,      #←訓練 200 次
                         batch_size=4,    #←每批次 4 筆資料
                         verbose=0)       #←安靜模式 (不顯示訊息)
    hv = history.history['val_mae']  #←取得驗證的歷史記錄
    idx = np.argmin(hv)       #←找出最佳驗證週期
    val = hv[idx]             #←取得最佳驗證的 mae 值
    u.plot(history.history,   #←繪製準確率的歷史線圖
           ('mae', 'val_mae'),
            f'k={i} Best val_mae at epoch={idx+1} val_mae={val:.3f}',
           ('Epoch','mae'), ylim=(0.03, 0.08)) #←限制 y 軸的數值範圍以方便觀看

    all_his_mae.append(history.history['mae'])     #←儲存 mae 歷史驗證資料
    all_his_val.append(history.history['val_mae']) #←儲存 val_mae 歷史驗證資料

avg_mae = np.mean(all_his_mae, axis=0)  #}←沿著第 0 軸 (k 折軸) 對每個 mae做平均,
avg_val = np.mean(all_his_val, axis=0)  #}  因此傳回 shape 為 (200,) 的向量
idx = np.argmin(avg_val)  #←找出最佳平均驗證結果的週期 (由 0 算起)
val =avg_val[idx]         #←取得最佳平均驗證的 val_mae 值
print(f'平均的最佳週期={idx+1}, val_mae={val:.3f}')  #←顯示最佳週期 (由 1 算起) 及其 val_mae 值
u.plot({'avg_mae': avg_mae, 'avg_val_mae': avg_val},  #←繪製歷史線圖
       ('avg_mae', 'avg_val_mae',),
       f'Best avg val_mae at epoch {idx+1} val_mae={val:.3f}',
       ('Epoch','mae'), ylim=(0.03, 0.08)) #←限制 y 軸的數值範圍以方便觀看

print(f'用所有的訓練資料重新訓練到第 {idx+1} 週期')
model.set_weights(wt)  #←還原初始權重 (效果等於重建模型, 以便重新訓練)
history =  model.fit(x_train, y_train,  #←用所有訓練資料進行訓練
                     epochs=idx+1,      #←訓練到最佳週期就停止
                     batch_size=4,      #←每批次 4 筆資料
                     verbose=0)         #←安靜模式 (不顯示訊息)
loss, mae = model.evaluate(x_test, y_test, verbose=0)  #←用測試資料評估成效
print(f'用測試資料評估的結果 mae={mae:.3f}')
