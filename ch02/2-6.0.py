import util2 as u  # 匯入自訂的工具模組

(x_train, x_test), (y_train, y_test) = u.mnist_data() # 載入預處理好的 MNIST 資料集
model = u.mnist_model()     # 取得新建立並編譯好的模型

model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)  # 訓練模型

ret = model.train_on_batch(x_train[0:64], y_train[0:64])  # 訓練一批資料
print('\ntrain:', ret)

ret = model.test_on_batch(x_test[0:32], y_test[0:32])  # 評估一批資料
print('\ntest:', ret)

ret = model.predict_on_batch(x_test[-3:]) # 預測一批資料 (測試樣本的最後 3 筆)
print('\npredict:\n', ret.round(1))     # 顯示預測出的類別索引
