# 設定卷積層不使用偏值
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()   
# 卷積層
cnn.add(Conv2D( 1, 
                (3, 3), 
                use_bias = False,     # 不使用偏差值
                input_shape=(5, 5, 1)))  
cnn.summary()



