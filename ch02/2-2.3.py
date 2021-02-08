from tensorflow.keras import metrics
import util2 as u      #←匯入自訂的工具模組

def top3_acc(y_true, y_pred):  #← 建立自訂評量函式             #↓指定 k 為 3
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

(x_train, x_test), (y_train, y_test) = u.mnist_data() #←載入預處理好的 MNIST 資料集
model = u.mnist_model()     #←取得新建立並編譯好的模型

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc', top3_acc])   #←使用自訂評量函式

model.fit(x_train, y_train, epochs=5, batch_size=128)  # 訓練模型
