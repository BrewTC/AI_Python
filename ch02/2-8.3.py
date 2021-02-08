from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.initializers import glorot_uniform

model = Sequential()
model.add(Dense(2, activation='relu', input_dim=1,
                kernel_initializer = glorot_uniform(seed=123)))

print(model.get_weights())
