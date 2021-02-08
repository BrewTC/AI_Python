from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate

a = Input(shape=(784,))
b = Input(shape=(784,))

shr  = Dense(512, activation='relu')    # 建立解析圖片用的共用層
out = concatenate([shr(a), shr(b)])     # 將 a,b 輸入到共用層, 再將其輸出串接起來

out = Dense(10, activation='relu')(out)    # 建立學習分類用的 Dense 層
out = Dense(1, activation='sigmoid')(out)  # 進行 2 元分類 (是否同數字)

model = Model([a, b], out)               # 建立模型
model.compile(optimizer='rmsprop',       # 編譯模型
              loss='binary_crossentropy', metrics=['acc'])

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 載入MNIST資料集
xa = train_images.reshape((60000, 28 * 28)).astype('float32') / 255  # 預處理圖片樣本
ya = train_labels        # 標籤不預處理, 因為稍後要用來比對是否為相同數字

idx = np.arange(40000)                      #}將 xa、ya 的前 4 萬筆隨機重排後,
np.random.shuffle(idx)                      #}連同後 2 萬筆一起另存到 xb、yb,
xb = np.concatenate((xa[idx], xa[40000:]))  #}這樣至少會有 2 萬筆以上的標籤為相同數字
yb = np.concatenate((ya[idx], ya[40000:]))  #}

y = np.where(ya==yb, 1.0, 0.0)   # 建立標籤：1 為是(相同數字), 0 為否

idx = np.arange(60000)               #} 再次將 xa/ya、xb/yb 同步隨機重排
np.random.shuffle(idx)               #}
xa, xb, y = xa[idx], xb[idx], y[idx] #}

print(f'訓練資料共 {len(y)} 筆, 其中有 {int(y.sum())} 筆為相同數字')

his = model.fit([xa, xb], y, validation_split=0.1,     #} 取 10% 做驗證, 訓練 20 週期
                epochs=20, batch_size=128, verbose=2)  #}

import util5 as u
u.plot(his.history, ('acc', 'val_acc'),     # 繪製訓練和驗證的準確率線圖
       'Training & Validating Acc', ('Epoch','Acc'))


