import util2 as u  # 匯入自訂工具模組, 並更名為 u

(x_train, x_test), (y_train, y_test) = u.mnist_data()  # 用 4 個變數接收傳回值
model = u.mnist_model()  # 取得新建立並編譯好的模型

model.fit(x_train, y_train, epochs=5, batch_size=128) # 用取得的資料與模型進行訓練


