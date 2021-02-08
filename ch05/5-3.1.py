from tensorflow.keras.layers import Conv2D, Input, add

inp = Input(shape=(256, 256, 3))

b = Conv2D(3, (3, 3), padding='same')(inp)
b = Conv2D(3, (3, 3), padding='same')(b)

out = add([b, inp])  # 將分支 b 和 inp 相加
