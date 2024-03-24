import matplotlib.pyplot as plt, japanize_matplotlib, keras, numpy as np
from keras import layers

from keras.datasets import cifar10 #スィーファーテンというカラー画像のデータセット
(x_train, y_train), (x_test, y_test)=cifar10.load_data()
x_train, x_test=x_train/255.0, x_test/255.0 #データの正規化 0~1の範囲にする
y_train, y_test=y_train.flatten(), y_test.flatten()

cat_train=x_train[np.where(y_train==3)]
cat_test=x_test[np.where(y_test==3)]
dog_train=x_train[np.where(y_train==5)]
dog_test=x_test[np.where(y_test==5)]
# print(len(cat_train), len(cat_test), len(dog_train), len(dog_test))

class_names=["ネコ", "イヌ"]
x_train=np.concatenate((cat_train, dog_train)) #concatenateで結合
x_test=np.concatenate((cat_test, dog_test))
y_train=np.concatenate((np.full(5000,0), np.full(5000,1))) #ネコを0 イヌを1とタグ付け (5000,0)で5000個の0に(5000,1)で5000個の1を追加
y_test=np.concatenate((np.full(1000,0), np.full(1000,1)))

np.random.seed(1) #乱数のシードを固定
np.random.shuffle(x_test) #データをシャッフル
np.random.seed(1) #全く同じ並びのシャッフルを作成
np.random.shuffle(y_test)


#! 学習データの水増し 過学習を防ぐ
from keras.preprocessing.image import ImageDataGenerator
data_gen=ImageDataGenerator(
    rotation_range=30, #30度の範囲でランダムに回転
    width_shift_range=0.1, #横方向にランダムにシフト
    height_shift_range=0.1, #縦方向にランダムにシフト
    zoom_range=0.1, #0.9~1.1倍の範囲でランダムに拡大縮小
    horizontal_flip=True, #水平方向にランダムに反転
)
g=data_gen.flow(x_train, y_train, shuffle=False) #flowで水増ししたデータを取得 
g_imgs1=[]
x_g, y_g=g.next() #nextで1つのデータを取得するたびに次のデータに移動 画像データとラベルデータ
g_imgs1.extend(x_g) #extendで1つのデータを取得するたびにリストに追加

g=data_gen.flow(x_train, y_train, shuffle=False) 
g_imgs2=[]
x_g, y_g=g.next()
g_imgs2.extend(x_g)


model=keras.models.Sequential()
model.add(layers.Conv2D(32, (5,5), activation="relu", input_shape=(32,32,3))) #32個の5×5のフィルター 32×32のカラー画像
model.add(layers.MaxPooling2D((2,2))) #2×2の最大プーリング
model.add(layers.Dropout(0.20)) #20%のニューロンを無効化
model.add(layers.Conv2D(64, (5,5), activation="relu"))
model.add(layers.MaxPooling2D((2,2)) )
model.add(layers.Dropout(0.20))
model.add(layers.Flatten()) #一次元配列に変換
model.add(layers.Dense(64, activation="relu")) #全結合層
model.add(layers.Dropout(0.20))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(2, activation="softmax")) #出力層
model.summary(line_length=120) #line_lengthで1行の文字数を120文字に増やす デフォルトは80文字

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history=model.fit(data_gen.flow(x_train, y_train), epochs=30, validation_data=(x_test, y_test)) #!data_gen.flow(x_train, y_train)で水増ししたデータを学習
test_loss, test_acc=model.evaluate(x_test, y_test) #評価
print(f"正解率:{test_acc:.2%}")


pre=model.predict(x_test)
plt.figure(figsize=(12,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([]) 
    plt.yticks([])
    plt.imshow(x_test[i]) 
    index=np.argmax(pre[i]) #一番高い確率のインデックス番号
    pct=pre[i][index] #一番高い確率の値
    ans=""
    if index!=y_test[i]: 
        ans="×--o["+class_names[y_test[i]]+"]"
    label=f"{class_names[index]}({pct:.2%}){ans}"
    plt.xlabel(label)
plt.show()



parameters=[
    ["正解率","accuracy", "val_accuracy"],
    ["損失(誤差)","loss", "val_loss"]
]

plt.figure(figsize=(10,4))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.title(parameters[i][0])
    plt.plot(history.history[parameters[i][1]], label="o-") #o-で丸と線を表示
    plt.plot(history.history[parameters[i][2]], label="o-")
    plt.xlabel("エポック(学習回数)")
    plt.legend(["学習", "検証"], loc="best")
    if i == 0:
        plt.ylim([0,1])
plt.show()
