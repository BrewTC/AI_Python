# 設定卷積核數量
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()   
# 卷積層
cnn.add( Conv2D(filters=32, 
                kernel_size=(3, 3), 
                input_shape=(5, 5, 1)))  
cnn.summary()