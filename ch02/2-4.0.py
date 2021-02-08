import util2 as u  # 匯入自訂的工具模組

(x_train, x_test), (y_train, y_test) = u.mnist_data() # 載入預處理好的 MNIST 資料集
model = u.mnist_model()     # 取得新建立並編譯好的模型

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc', 'mse'])  #←使用 2 個評量準則

model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)  # 訓練模型
                                                       # ↑ 安靜模式
t_loss, t_acc, t_mse = model.evaluate(x_test, y_test) #←使用測試樣本及標籤來評估普適能力
print('對測試資料集的損失值：', t_loss)
print('對測試資料集的準確率：', t_acc)
print('對測試資料集的均方差：', t_mse)