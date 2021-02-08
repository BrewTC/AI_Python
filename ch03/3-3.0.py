# 使用展平層來將特徵圖展平層 1D 向量
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

cnn = Sequential()
# 卷積層
cnn.add( Conv2D(filters=8, 		                # 8 個神經元, 輸出 8 張特徵圖
			    kernel_size=(3, 3),	            # 卷積核尺寸 3x3
                padding = 'same',	                # 使用填補法
                input_shape=(100, 100, 1)))	    # 輸入圖片尺寸為 100x100
# 池化層
cnn.add( MaxPooling2D(pool_size=(2, 2),	        # 檢視視窗 2x2
				      strides = 2))               # 向右向下步長 = 2

# 展平層
cnn.add(Flatten())                               # 將特徵圖拉平
cnn.summary()
