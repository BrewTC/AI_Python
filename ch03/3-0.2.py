from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # 卷積層
cnn.add(MaxPooling2D((2, 2)))                                            # 池化層
cnn.add(Conv2D(64, (3, 3), activation='relu'))                           # 卷積層
cnn.add(MaxPooling2D((2, 2)))                                            # 池化層
cnn.add(Conv2D(64, (3, 3), activation='relu'))                           # 卷積層
cnn.add(Flatten())                                                       # 展平層
cnn.add(Dense(64, activation='relu'))                                    # 密集層
cnn.add(Dense(10, activation='softmax'))                                 # 密集層

cnn.summary()