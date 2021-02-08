# 使用卷積層進行降維
from tensorflow.keras import models, layers


cnn = models.Sequential()
# 第一層卷積層
cnn.add(layers.Conv2D(  filters=32,
                        kernel_size=(2, 2),
                        padding = 'same',
                        input_shape=(100, 100, 1),
                        ))

# 第二層卷積層
cnn.add(layers.Conv2D(  filters=32,
                        kernel_size=(2, 2),
                        strides = 2,        # 卷積步長設定為 2
                        padding = 'same',
                        ))


cnn.summary()

i = 3
