import keras,numpy as np, matplotlib.pyplot as plt, japanize_matplotlib,sklearn
from keras import layers

#! XOR回路を作る 2つの値が違うなら1,同じなら0
input_data=[
    [0,0],
    [1,0],
    [0,1],
    [1,1]
]
xor_data=[0,1,1,0]
x_train=x_test=np.array(input_data) #kerasの入力はnumpyのarrayに変換する必要がある x:問題 y:答え
y_train=y_test=np.array(xor_data)

#モデルを作る
model=keras.models.Sequential() #Sequential:層を積み重ねる
#Dense:全結合層(次の層にあるすべてのニューロンと結合する層)  8:ニューロンの数 activation:活性化関数 input_dim:最初の層なので入力の次元数
model.add(layers.Dense(8, activation="relu",input_dim=2)) 
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(2, activation="softmax")) #最後の層は分類問題なのでsoftmax関数で出力値を確率に変換する
model.summary() #モデルの概要を表示 
#! 3つの全結合層(Dense)かでき、重みは線に付くから8×2個、閾値はニューロンに付くから8個で計24個のパラメータがある 2層目も8×8+8=72個 3層目も8×2+2=18個

model.compile(
    optimizer="adam", #最適化アルゴリズム adam:学習率を自動で調整する
    loss="sparse_categorical_crossentropy", #多クラス分類問題の損失関数 交差エントロピー誤差
    metrics=["accuracy"] #正解率
)
history=model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test)) #epochs:学習回数 validation_data:検証用データ
test_loss, test_acc=model.evaluate(x_test, y_test) #評価
print(f"正解率:{test_acc:.2%}")


#! 予測
pre=model.predict(x_test)
print(pre) #それぞれの確率を表示 0,1の確率が4行2列で表示される
for i in range(4):
    index=np.argmax(pre[i]) #最大値のインデックスを取得 一番高い確率の番号を取得
    print(f"{i+1}番目のデータは{index}です。")


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