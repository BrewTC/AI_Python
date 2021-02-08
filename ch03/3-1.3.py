# 將偏值初始值設定為 3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import Constant

cnn = Sequential()   
# 卷積層
cnn.add(Conv2D( 1, 
                (3, 3), 
                bias_initializer = Constant(value=3),
            #   bias_initializer = 'constant',  # 別名字串用法
                input_shape=(5, 5, 1)))  
print(cnn.get_weights())

