# 建立一個接收彩色圖片的卷積層
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()
# 卷積層
cnn.add( Conv2D( 1,
                (3, 3),
                input_shape=(32, 32, 3)))
cnn.summary()