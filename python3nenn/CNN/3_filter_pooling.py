import numpy as np
import matplotlib.pyplot as plt

vdata=[]
hdata=[]
vpool=[]
hpool=[]

n0 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,4,9,9,9,9,9,9,4,0,0],
    [0,0,9,9,9,9,9,9,9,9,0,0],
    [0,0,9,9,4,0,0,4,9,9,0,0],
    [0,0,9,9,0,0,0,0,9,9,0,0],
    [0,0,9,9,0,0,0,0,9,9,0,0],
    [0,0,9,9,0,0,0,0,9,9,0,0],
    [0,0,9,9,0,0,0,0,9,9,0,0],
    [0,0,9,9,4,0,0,4,9,9,0,0],
    [0,0,9,9,9,9,9,9,9,9,0,0],
    [0,0,4,9,9,9,9,9,9,4,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
n1 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,4,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,0,9,7,0,0,0,0,0],
    [0,0,0,0,4,9,7,4,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
n2 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,4,9,9,9,9,9,9,4,0,0],
    [0,0,9,9,9,9,9,9,9,9,0,0],
    [0,0,0,0,0,0,0,4,9,9,0,0],
    [0,0,0,0,0,0,0,0,9,9,0,0],
    [0,0,0,0,4,9,9,9,9,9,0,0],
    [0,0,0,0,4,9,9,9,9,9,0,0],
    [0,0,0,0,0,0,0,0,9,9,0,0],
    [0,0,0,0,0,0,0,4,9,9,0,0],
    [0,0,9,9,9,9,9,9,9,9,0,0],
    [0,0,4,9,9,9,9,9,9,4,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
ndata = [n0, n1, n2]

filter_vertical = np.array([
    [-2.0, 1.0, 1.0],
    [-2.0, 1.0, 1.0],
    [-2.0, 1.0, 1.0]]) #縦の線を検出するフィルタ
filter_horizontal= np.array([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-2.0, -2.0, -2.0]]) #横の線を検出するフィルタ

def convo_img(num_img, filter):
    num_x, num_y = len(num_img), len(num_img[0])
    img=np.zeros((num_x, num_y))
    for i in range(num_x-3+1):
        for j in range(num_y-3+1):
            img[i][j]=np.sum(num_img[i:i+3, j:j+3]*filter) #フィルターをかける
    return img

def pooling_img(num_img, num):
    img=[]
    num_img=np.array(num_img)
    num_x, num_y = len(num_img), len(num_img[0])
    for i in range(0, num_x, num):
        row=[]
        for j in range(0, num_y, num):
            row.append(np.max(num_img[i:i+num, j:j+num]))
        img.append(row) #最大値を取り出す
    return img

def cnn_test(data, num, size):
    for idx in range(num):
        vdata.append(convo_img(data[idx], filter_vertical))
        hdata.append(convo_img(data[idx], filter_horizontal))
        vpool.append(pooling_img(vdata[idx], size))
        hpool.append(pooling_img(hdata[idx], size))

    plt.figure(figsize=(12,8))
    for idx in range(num):
        for i in range(5):
            plt.subplot(num, 5, idx*5+i+1)
            if i == 0:
                plt.imshow(data[idx], cmap="Greys")
            if i == 1:
                plt.imshow(vdata[idx], cmap="Blues")
                plt.xlabel("Vertical")
            if i == 2:
                plt.imshow(vpool[idx], cmap="Blues")
            if i == 3:
                plt.imshow(hdata[idx], cmap="Blues")
                plt.xlabel("Horizontal")
            if i == 4:
                plt.imshow(hpool[idx], cmap="Blues")
            plt.xticks([])
            plt.yticks([])
    plt.show()

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()
x_train, x_test=x_train/255.0, x_test/255.0 #データの正規化 0~1の範囲にする


cnn_test(x_test, 5, 5)
#2つ目が縦線フィルタの畳み込み層に通した画像、3つ目がそれをプーリング層に通した画像、4つ目が横線フィルタ、5つ目が横線プーリング