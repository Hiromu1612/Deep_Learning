import matplotlib.pyplot as plt, japanize_matplotlib, keras, numpy as np
from keras import layers

from keras.datasets import cifar10 #スィーファーテンというカラー画像のデータセット
(x_train, y_train), (x_test, y_test)=cifar10.load_data()
x_train, x_test=x_train/255.0, x_test/255.0 #データの正規化 0~1の範囲にする

# print(x_train.shape, x_test.shape) #学習データとテストデータの形状を表示 32×32のカラー画像が50000枚と10000枚
class_names=["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "蛙", "馬", "船", "トラック"]
# def disp_data(xdata, ydata):
#     plt.figure(figsize=(12,10))
#     for i in range(20):
#         plt.subplot(4,5,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(xdata[i], cmap="Greys")
#         plt.xlabel(class_names[ydata[i][0]]) #[0]をつけることでリストから数値を取り出す 今回はリストの中にリストがあるため
#     plt.show()

# disp_data(x_train, y_train)

model=keras.models.Sequential()
model.add(layers.Flatten(input_shape=(32,32,3))) #カラー画像のため3次元 3072個の一次元配列に変換
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history=model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
test_loss, test_acc=model.evaluate(x_test, y_test) #評価
print(f"正解率:{test_acc:.2%}")

parameters=[
    ["正解率","accuracy", "val_accuracy"],
    ["損失(誤差)","loss", "val_loss"]
]

pre=model.predict(x_test)
plt.figure(figsize=(12,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([]) #x軸の目盛りを消す
    plt.yticks([])
    plt.imshow(x_test[i]) #画像を表示
    index=np.argmax(pre[i]) #最大値のインデックスを取得 一番高い確率の番号を取得
    pct=pre[i][index] #一番高い確率を取得
    ans=""
    if index!=y_test[i][0]: #等しくない場合
        ans="×--o["+class_names[y_test[i][0]]+"]"
    label=f"{class_names[index]}({pct:.2%}){ans}" #予測結果と正解率を表示
    plt.xlabel(label)
plt.show()




plt.figure(figsize=(10,5))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.plot(history.history[parameters[i][1]], label="学習")
    plt.plot(history.history[parameters[i][2]], label="検証")
    plt.title(parameters[i][0])
    plt.xlabel("エポック(学習回数)")
    plt.ylabel(parameters[i][0])
    plt.legend(["学習","検証"], loc="best") #loc="best"で最適な位置に凡例を表示
    if i == 0:
        plt.ylim(0,1) #y軸の範囲を0~1にする
plt.show()