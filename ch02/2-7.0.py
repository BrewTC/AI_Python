from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l1(),
                activity_regularizer=regularizers.l1_l2(l1=0.002, l2=0.001)))

from tensorflow.keras import backend as K  # 匯入後端函式庫介面

def my_l1(weight_matrix):   # 自訂常規化函式
    return 0.03 * K.sum( K.abs(weight_matrix) )  # 傳回以強度 0.03 計算的 L1 懲罰值

model = Sequential()
model.add(Dense(64, input_dim=64,
                kernel_regularizer=my_l1))  # 使用自訂的常規化函式

