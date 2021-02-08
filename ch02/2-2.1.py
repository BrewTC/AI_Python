import util2 as u      #←匯入自訂的工具模組

(x_train, x_test), (y_train, y_test) = u.mnist_data() #←載入預處理好的 MNIST 資料集
model = u.mnist_model()     #←取得新建立並編譯好的模型

from tensorflow.keras import metrics      #← 匯入評量準則模組

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc', 'mse', metrics.top_k_categorical_accuracy])
                              #↑ 同時指定 3 個評量準則, 其中第 2 個為損失函數

model.fit(x_train, y_train, epochs=5, batch_size=128) # 訓練模型