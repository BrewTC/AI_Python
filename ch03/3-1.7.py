# 啟用填補法
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()   
# 卷積層
cnn.add( Conv2D( 1, 
                (3, 3), 
                padding = 'same',  # 啟用填補法
                input_shape=(5, 5, 1)))  
cnn.summary()