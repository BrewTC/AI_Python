
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda

A = Input((4,))
B = Input((5,))

lmb = Lambda(lambda x: [x/2, x*2]) # Lambda 層
C,D = lmb(A)  # 輸入 A, 輸出 C、D
E,F = lmb(B)  # 輸入 B, 輸出 E、F

C = Dense(6)(C)   #}
D = Dense(7)(D)   #} 分別將 C,D,E,F 都
E = Dense(8)(E)   #} 連到一個 Dense 層
F = Dense(9)(F)   #}

model = Model([A, B], [C, D, E, F]) # 建立模型
model.summary()

