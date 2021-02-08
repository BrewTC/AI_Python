from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') /255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')  /255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()                      # 建立 CNN 模型
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',         # 編譯模型
              optimizer='Adam', metrics=['acc'])

model.fit(x_train, y_train, batch_size=128, epochs=12) # 訓練模型

score = model.evaluate(x_test, y_test, verbose=0)      # 評估成效
print('對測試資料集的準確率：', score[1])

model.save('模型_MNIST_CNN.h5')     # 將模型存檔, 以供稍後使用
