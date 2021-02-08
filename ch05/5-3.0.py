from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate

inp = Input(shape=(256, 256, 3))  # 輸入層 (256x256 的 3 通道圖片)

b1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp) # 第 1 分支
b1 = Conv2D(64, (3, 3), padding='same', activation='relu')(b1)

b2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp) # 第 2 分支
b2 = Conv2D(64, (5, 5), padding='same', activation='relu')(b2)

b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp)  # 第 3 分支
b3 = Conv2D(64, (1, 1), padding='same', activation='relu')(b3)

out = concatenate([b1, b2, b3], axis=1)  # 將 3 個分支串接起來

