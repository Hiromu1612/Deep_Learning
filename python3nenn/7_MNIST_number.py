import matplotlib.pyplot as plt, japanize_matplotlib, keras, numpy as np
from keras import layers

from keras.datasets import mnist #mnistは数字のデータセット
(x_train, y_train), (x_test, y_test)=mnist.load_data() #データの読み込み
x_train, x_test=x_train/255.0, x_test/255.0 #データの正規化 0~1の範囲にする
print(x_train.shape, x_test.shape) #学習データとテストデータの形状を表示　28×28の画像が60000枚と10000枚

# def disp_data(xdata,ydata):
#     plt.figure(figsize=(12,10))
#     for i in range(20):
#         plt.subplot(4,5,i+1)
#         plt.xticks([]) #x軸の目盛りを消す
#         plt.yticks([])
#         plt.imshow(xdata[i], cmap="Greys") #画像を表示
#         plt.xlabel(ydata[i])
#     plt.show()
    
# disp_data(x_train, y_train) #学習データを表示
# disp_data(x_test, y_test) #テストデータを表示

model=keras.models.Sequential() #Sequential:層を積み重ねる
model.add(layers.Flatten(input_shape=(28,28))) #入力層　Flatten:28×28の画像を1次元に変換して画像を読みこむ
model.add(layers.Dense(128, activation="relu")) #中間層　Dense:全結合層(次の層にあるすべてのニューロンと結合する層) 128:ニューロンの数 activation:活性化関数
model.add(layers.Dense(10, activation="softmax")) #出力層　10:出力の数(0~9) softmax関数で出力値を確率に変換する
model.summary() #モデルの概要を表示

model.compile(
    optimizer="adam", #最適化アルゴリズム adam:学習率を自動で調整する
    loss="sparse_categorical_crossentropy", #多クラス分類問題の損失関数 交差エントロピー誤差
    metrics=["accuracy"] #正解率
)
#!今回はデータが豊富で、ニューロン数も多いため学習回数は少なめで良い
history=model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test)) #epochs:学習回数 validation_data:検証用データ
test_loss, test_acc=model.evaluate(x_test, y_test) #評価
print(f"正解率:{test_acc:.2%}")


#! 予測
pre=model.predict(x_test)
plt.figure(figsize=(12,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([]) #x軸の目盛りを消す
    plt.yticks([])
    plt.imshow(x_test[i], cmap="Greys") #画像を表示
    index=np.argmax(pre[i]) #最大値のインデックスを取得 一番高い確率の番号を取得
    pct=pre[i][index] #一番高い確率を取得
    ans=""
    if index!=y_test[i]: #等しくない場合
        ans="×--o["+str(y_test[i])+"]"
    label=f"{index}({pct:.2%}){ans}" #予測結果と正解率を表示
    plt.xlabel(label)
plt.show()

#! 学習の様子を可視化
parameters=[
    ["正解率","accuracy", "val_accuracy"],
    ["損失(誤差)","loss", "val_loss"]
]
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