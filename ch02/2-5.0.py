import util2 as u  # 匯入自訂的工具模組

(x_train, x_test), (y_train, y_test) = u.mnist_data() # 載入預處理好的 MNIST 資料集
model = u.mnist_model()     # 取得新建立並編譯好的模型

model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)  # 訓練模型

predict = model.predict(x_test[0:3])  # ←取前 3 筆測試樣本做預測
print(predict.round(1))   # ←顯示四捨五入到小數 1 位的預測結果

predict = model.predict_classes(x_test[0:3])  # ←取前 3 筆測試樣本做預測
print(predict)            # ←顯示預測出的類別索引
