
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, concatenate

a = Input(shape=(28, 28, 1))  #} 建立 2 個輸入層, 輸入 shape 均為 (28, 28, 1)
b = Input(shape=(28, 28, 1))  #}

cnn = load_model('模型_MNIST_CNN.h5')    # 載入已訓練好的 CNN 模型
cnn.trainable = False                   # 將模型設為不可訓練 (鎖住權重)

out = concatenate([cnn(a), cnn(b)])  # 將 a,b 輸入到 CNN 模型層, 並將輸出串接起來
out = Dense(128, activation='relu')(out)    # 建立學習分類用的 Dense 層
out = Dense(1, activation='sigmoid')(out)   # 進行 2 元分類 (是否同數字)

model = Model([a, b], out)               # 建立模型
model.compile(optimizer='rmsprop',       # 編譯模型
              loss='binary_crossentropy', metrics=['acc'])

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 載入MNIST資料集
xa = train_images.reshape((60000,784)).astype('float32') / 255  # 預處理圖片樣本
ya = train_labels        # 標籤不預處理, 因為稍後要用來比對是否為相同數字

idx = np.arange(40000)                      #}將 xa、ya 的前 4 萬筆隨機重排後,
np.random.shuffle(idx)                      #}連同後 2 萬筆一起另存到 xb、yb,
xb = np.concatenate((xa[idx], xa[40000:]))  #}這樣至少會有 2 萬筆以上的標籤為相同數字
yb = np.concatenate((ya[idx], ya[40000:]))  #}

y = np.where(ya==yb, 1.0, 0.0)   # 建立標籤：1 為是(相同數字), 0 為否

idx = np.arange(60000)               #} 再次將 xa/ya、xb/yb 同步隨機重排
np.random.shuffle(idx)               #}
xa, xb, y = xa[idx], xb[idx], y[idx] #}

xa = xa.reshape((60000,28,28,1))  #} 將樣本改為符合 CNN 輸入的 shape
xb = xb.reshape((60000,28,28,1))  #}
n = 2000                             # 設定只取前 2000 筆來做訓練
print(f'訓練資料共 {len(y[:n])} 筆, 其中有 {int(y[:n].sum())} 筆為相同數字')
                #         ↑                       ↑
                # 只取前 n 筆做訓練
                #   ↓       ↓     ↓
his = model.fit([xa[:n], xb[:n]], y[:n], validation_split=0.1,   #} 取 10% 做驗證, 訓練 20 週期
                epochs=20, batch_size=128, verbose=2)  #}

import util5 as u
u.plot(his.history, ('acc', 'val_acc'),     # 繪製訓練和驗證的準確率線圖
       'Training & Validating Acc', ('Epoch','Acc'))

                  # 將剩下的資料拿來評估成效
                  #       ↓                       ↓
print(f'測試資料共 {len(y[n:])} 筆, 其中有 {int(y[n:].sum())} 筆為相同數字')
score = model.evaluate([xa[n:], xb[n:]], y[n:], verbose=0)      # 評估成效
print('對測試資料集的準確率：', score[1])