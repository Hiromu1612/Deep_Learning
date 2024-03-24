from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim) #軸(次元)の数 0次元テンソル:スカラー 1:ベクトル 2:行列
print(train_images.shape) #形状 各軸の次元数
print(train_images.dtype) #データ型 uint8 0~255の整数

import matplotlib.pyplot as plt
digit=train_images[4]
plt.imshow(digit, cmap=plt.cm.binary) #binaryは白黒
plt.show()