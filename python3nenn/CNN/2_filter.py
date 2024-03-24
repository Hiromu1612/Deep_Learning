import numpy as np
import matplotlib.pyplot as plt

filter_vertical = np.array([
    [-2.0, 1.0, 1.0],
    [-2.0, 1.0, 1.0],
    [-2.0, 1.0, 1.0]]) #縦の線を検出するフィルタ
filter_horizontal= np.array([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-2.0, -2.0, -2.0]]) #横の線を検出するフィルタ

for i in range(2):
    plt.subplot(1,2,i+1)
    if i == 0:
        plt.imshow(filter_vertical, cmap="Grays") #Bluesは色の名前
        plt.xlabel("Vertical")
    if i == 1:
        plt.imshow(filter_horizontal, cmap="Grays")
        plt.xlabel("Horizontal")
    plt.xticks([])
    plt.yticks([])
plt.show()