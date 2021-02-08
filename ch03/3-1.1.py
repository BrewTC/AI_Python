# 比較密集層與卷積層的權重數目

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D


# 密集層
dense = Sequential()
dense.add(Dense(1,                  # 一個神經元
          input_dim=28*28*1)) # 輸入圖片的所有像素數量
print('\n---- ↓ 查看密集層資訊 ↓ ----')
dense.summary()

# 卷積層
cnn = Sequential()
cnn.add(Conv2D( 1,		# 請 1 個小朋友幫忙繪製特徵圖
                (3, 3),	# 卷積核尺寸
                input_shape=(28, 28, 1)))	# 輸入圖片的尺寸
print('\n---- ↓ 查看卷積層資訊 ↓ ----')
cnn.summary()
