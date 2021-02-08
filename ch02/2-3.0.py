import util2 as u  # 匯入自訂的工具模組

(x_train, x_test), (y_train, y_test) = u.mnist_data() #←載入預處理好的 MNIST 資料集
model = u.mnist_model()     #←取得新建立並編譯好的模型

history = model.fit(x_train, y_train, epochs=5, batch_size=128,
                    validation_split=0.2,  # 將測試資資切出 20% 做為驗證用
                    verbose=2)             # 精簡模式 (無進度條)

print(history.history)  # 顯示訓練的歷史資料

u.plot(history.history, ('loss','val_loss'),  # 繪製損失值歷史線圖
       'Training & Validation loss',
       ('Epochs', 'Loss'))

u.plot(history.history, ('acc','val_acc'),    # 繪製準確度歷史線圖
       'Training & Validation accuracy',
       ('Epochs', 'Accuracy'))