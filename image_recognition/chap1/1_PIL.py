from PIL import Image
import numpy as np
img_gray=Image.open(r"C:\Users\1612h\Deep_Learning\img_recognition\chap1\coffee.jpg")
img_gray.show(img_gray)

# import cv2
# img_gray=cv2.imread(r"C:\Users\1612h\Deep_Learning\img_recognition\chap1\coffee.jpg") #cv2.IMREAD_GRAYSCALE:グレースケールで読み込む
# cv2.imshow("img_gray", img_gray) #img_grayという名前で表示
# cv2.waitKey(0) #キー入力待ち
# cv2.destroyAllWindows() #すべてのウィンドウを閉じる


print(np.array(img_gray)) #画像のピクセル値を表示
print(np.array(img_gray).shape) #(高さ,幅,チャンネル数) チャネル数はRGBなら3,グレースケールなら1