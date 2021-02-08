import util2 as u  # 匯入自訂的工具模組

#####↓自訂生成器↓#####
def trn_gen():
    while True:
        for i in range(0, 50000, 500): #←前 50000 筆訓練資料拿來訓練, 每批次 500 筆
            yield (x_train[i:i+500], y_train[i:i+500]) #←每次傳回 (500筆樣本, 500筆標籤)

def val_gen():
    while True:
        for i in range(50000, 60000, 1000): #←後 10000 筆訓練資料拿來驗證, 每批次 1000 筆
            yield (x_train[i:i+1000], y_train[i:i+1000]) #←每次傳回 (1000筆樣本, 1000筆標籤)

def eva_gen():
    while True:
        for i in range(0, 10000, 2000):  #← 10000 筆測試資料拿來評量成效, 每批次 2000 筆
            yield (x_test[i:i+2000], y_test[i:i+2000])#←每次傳回 (2000筆樣本, 2000筆標籤)

def prd_gen():
    while True:
        for i in range(0, 10000, 2):  #← 10000 筆測試資料拿來評量成效, 每批次 2 筆
            yield (x_test[i:i+2], y_test[i:i+2])#←每次傳回 (2筆樣本, 2筆標籤)

#####↓主程式↓#####
(x_train, x_test), (y_train, y_test) = u.mnist_data() #←載入預處理好的 MNIST 資料集
model = u.mnist_model()     #←取得新建立並編譯好的模型
                                     #↓每週期訓練 5000/500=100 批次
model.fit(trn_gen(), steps_per_epoch=100,            #↓驗證 10000/1000=10 批次
          validation_data=val_gen(), validation_steps=10,
          epochs=5)

print('評估成效：')            #↓評估 10000/2000=5 批次
model.evaluate(eva_gen(), steps=5)

print('預測前 4 筆測試資料的答案：')   #↓預測 2 批次的測試資料 (每批次只有 2 筆)
ret = model.predict(prd_gen(), steps=2)
print(ret.round(1))