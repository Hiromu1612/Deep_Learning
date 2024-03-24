x1=[0,1,0,1]
x2=[0,0,1,1]

#AND回路
def test(x1,x2):
    if x1==1 and x2==1:
        return 1
    else:
        return 0
    
def result(func):
    for i in range(4):
        Y=func(x1[i],x2[i]) #funcにx1[i],x2[i]を渡して、結果をYに代入
        print(f"{x1[i]}, {x2[i]}={Y}") #f文字列は{数値、変数}で変数を文字列に埋め込む

result(test) #resultにtestを渡して実行 


#! AND回路のパーセプトロン
def and_test(x1,x2):
    w1,w2,threshold=0.5,0.5,0.7 #閾値 threshold
    ans=x1*w1+x2*w2
    if ans > threshold:
        return 1
    else:
        return 0
result(and_test) 

#分類の状態を可視化
import numpy as np
import matplotlib.pyplot as plt

def fillscolors(data):
    return "#ffc2c2" if data > 0 else "#c6dcec"
def dotscolors(data):
    return "#ff0e0e" if data > 0 else "#1f77b4"

def plot_perceptron(func, X1, X2):
    plt.figure(figsize=(6, 6))
    #meshgrid関数で格子点を生成 背景にも点をプロットする linespace関数で-0.25から1.25までの範囲を200分割
    XX, YY = np.meshgrid(
        np.linspace(-0.25, 1.25, 200),
        np.linspace(-0.25, 1.25, 200))
    XX = np.array(XX).flatten()
    YY = np.array(YY).flatten()
    fills = []
    colors = []
    for i in range(len(XX)):
        fills.append(func(XX[i], YY[i]))
        colors.append(fillscolors(fills[i]))
    plt.scatter(XX, YY, c=colors)

    dots = []
    colors = []
    for i in range(len(X1)):
        dots.append(func(X1[i], X2[i]))
        colors.append(dotscolors(dots[i]))
    plt.scatter(X1, X2, c=colors) #X1,X2の点をプロット
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

plot_perceptron(and_test, x1, x2)