import util2 as u      #←匯入自訂的工具模組
from tensorflow.keras import backend as K  #←匯入 Keras 後端函式庫介面

(x_train, x_test), (y_train, y_test) = u.mnist_data() #←載入預處理好的 MNIST 資料集
model = u.mnist_model()     #←取得新建立並編譯好的模型

def my_mse(y_true, y_pred):   # 自訂損失函數
    return K.mean(K.square(y_pred - y_true), axis=-1)
          #↑ Keras 已將後端的函式都包裝起來以方便使用

model.compile(optimizer='rmsprop',
              loss=my_mse,    #← 使用自訂的損失函數
              metrics=['acc'])

# 訓練模型
history = model.fit(x_train, y_train, epochs=5, batch_size=128)
print('評估成效：', model.evaluate(x_test, y_test, verbose=0))