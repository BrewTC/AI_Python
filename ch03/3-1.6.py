# 設定卷積步長
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()   
# 卷積層
cnn.add( Conv2D(1, (3, 3), 
                strides=(2,1),        # 右移步長 = 2、下移步長 = 1
                input_shape=(28, 28, 1)))  
cnn.summary()