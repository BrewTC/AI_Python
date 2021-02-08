from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

network_cifar10 = Sequential()
network_cifar10.add(Dense(512, activation='relu', input_dim=32*32*3))    # 密集層
network_cifar10.add(Dense(10, activation='softmax'))                     # 密集層

network_cifar10.summary()
