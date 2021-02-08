# -- 載入 cifar10 資料集 -- #
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# -- 查看資料的 shape -- #
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# -- 查看標籤的 shape -- #
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# -- 查看標籤的內容 -- #
print(y_train[0: 10])   # 查看前 10 個訓練標籤

# -- 進行 min-max normalization -- #
# min-max normalization 前
print(x_train[0][0][0])

# 進行 min-max normalization...
x_train_norm = x_train.astype('float32') / 255	# 每個像素值除以 255
x_test_norm = x_test.astype('float32') / 255  	# 每個像素值除以 255

# -- min-max normalization 後 -- #
print(x_train_norm[0][0][0])



# -- 將數字標籤進行 One-hot 編碼 -- #
from tensorflow.keras import utils
# 轉換前
print(y_train[0])

# 進行 One-hot 編碼轉換...
y_train_onehot = utils.to_categorical(y_train, 10) # 將訓練標籤進行 One-hot 編碼
y_test_onehot = utils.to_categorical(y_test, 10)	 # 將測試標籤進行 One-hot 編碼

# 轉換後
print(y_train_onehot[0])

# -- 建立 CNN 神經網路架構 -- #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                input_shape=(32, 32, 3)))  					# 卷積層 (輸入)
cnn.add(Dropout(0.25))           							# Dropout
cnn.add(MaxPooling2D((2, 2)))                        			# 池化層

cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))   	# 卷積層
cnn.add(Dropout(0.25))                              			# Dropout 層
cnn.add(MaxPooling2D((2, 2)))                             		# 池化層

cnn.add(Flatten())                                    			# 展平層
cnn.add(Dropout(0.25))                                   		# Dropout
cnn.add(Dense(1024, activation='relu'))                      	# 密集層
cnn.add(Dropout(0.25))                                     	# Dropout
cnn.add(Dense(10, activation='softmax'))                    		# 密集層 (輸出分類)

# -- 神經網路的訓練配置 -- #
cnn.compile(loss='categorical_crossentropy',	# 損失函數
              optimizer='adam',				    # adam優化器
              metrics=['acc'])			    # 以準確度做為訓練指標


# -- 進行訓練 -- #
history = cnn.fit(x=x_train_norm,   	# 訓練資料
	  		    y=y_train_onehot,		# 訓練標籤
      		    batch_size=128,		# 每個批次用 128 筆資料進行訓練
      		    epochs=20,			# 20 個訓練週期 (次數)
      		    validation_split = 0.1, 	# 拿出訓練資料的 10% 做為驗證資料
			    )

# --  繪製圖表 -- #
import util3 as u    	# 匯入自訂的繪圖工具模組

u.plot( history.history,   # 繪製準確率與驗證準確度的歷史線圖
        ('acc', 'val_acc'),
        'Training & Vaildation Acc',
        ('Epoch','Acc'),
        )

u.plot( history.history,   #  繪製損失及驗證損失的歷史線圖
        ('loss', 'val_loss'),
        'Training & Vaildation Loss',
        ('Epoch','Loss'),
        )

# --  儲存模型 -- #
cnn.save('CNN_Model.h5')

# -- 儲存模型權重-- #
cnn.save_weights('CNN_weights.h5')

# -- 載入模型 -- #
from tensorflow.keras.models import load_model

old_cnn = load_model('CNN_Model.h5')
print('載入模型成功')

# -- 使用測試資料評估神經網路 -- #
test_loss, test_val = cnn.evaluate(x_test_norm, y_test_onehot)
print('測試資料損失值:', test_loss)
print('測試資料準確度:', test_val)

# -- 查看神經網路的預測結果 -- #
predict_prop = cnn.predict(x_test_norm)
print('第一筆測試資料的預測機率', predict_prop[0])

# -- 查看測試資料的第 1 張圖片 -- #
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(2, 2)
plt.imshow(x_test[0])
plt.show()

# -- 查看測試資料的第 2 張圖片 -- #
fig = plt.gcf()
fig.set_size_inches(2, 2)
plt.imshow(x_test[1])
plt.show()

# -- 直接預測數字標籤 -- #
predict_class = cnn.predict_classes(x_test_norm)
print('前 10 筆預測標籤:', predict_class[: 10])
print('前 10 筆原始標籤:', y_test[: 10].reshape(10))

