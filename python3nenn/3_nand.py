x1=[0,1,0,1]
x2=[0,0,1,1]

def result(func):
    for i in range(4):
        Y=func(x1[i],x2[i]) #funcにx1[i],x2[i]を渡して、結果をYに代入
        print(f"{x1[i]}, {x2[i]}={Y}") #f文字列は{数値、変数}で変数を文字列に埋め込む

#! NAND回路のパーセプトロン -0.5,-0.8にするだけ 両方とも1の時だけ0を返す
def nand_test(x1,x2):
    w1,w2,threshold=-0.5,-0.5,-0.8 #閾値 threshold
    ans=x1*w1+x2*w2
    if ans > threshold:
        return 1
    else:
        return 0
result(nand_test) 