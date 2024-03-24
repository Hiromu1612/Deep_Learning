"""リスト1-2"""
import time
import keras
from keras import layers
from keras.datasets import mnist

start_time=time.time()

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5,
          validation_data=(x_test, y_test))

"""リスト1-3"""

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(x_test[0], cmap="Greys")
plt.show()
pre = model.predict(x_test)
index = np.argmax(pre[0])
print(f"この画像は「{index}」です。 ")

end_time=time.time()
print(f"経過時間：{end_time-start_time}秒")