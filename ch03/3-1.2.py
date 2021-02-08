# 更換權重初始器
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import tensorflow.keras

cnn = Sequential()
# 卷積層
cnn.add(Conv2D( 1,
                (3, 3),
                kernel_initializer = tensorflow.keras.initializers.Ones(),
                # kernel_initializer = 'Ones',
                #   kernel_initializer = 'ones',
                #   kernel_initializer = 'one',
                input_shape=(5, 5, 1)))

print(cnn.get_weights())
