# 使用最大池化層
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

cnn = Sequential()   
# 卷積層
cnn.add( Conv2D(filters=32, 
                kernel_size=(3, 3), 
                padding = 'same',
                input_shape=(100, 100, 1),
                ))

# 池化層
cnn.add( MaxPooling2D(pool_size=(2, 2),
                      strides = 2,
                    ))
                                  
cnn.summary()