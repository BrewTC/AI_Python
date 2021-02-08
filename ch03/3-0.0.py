from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

network_nmist = Sequential()
network_nmist.add(Dense(512, activation='relu', input_dim=28*28))    # 密集層
network_nmist.add(Dense(10, activation='softmax'))                   # 密集層

network_nmist.summary()

