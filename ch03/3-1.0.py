from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()
# 卷積層
cnn.add(Conv2D( 1,
                (3, 3),
                input_shape=(7, 7, 1)))
cnn.summary()
# print(cnn.get_weights())