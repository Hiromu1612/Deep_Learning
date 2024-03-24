from keras.applications import VGG16
model=VGG16()
model.summary()

from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# testimg = load_img(r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\test.jpg", target_size=(224,224))
# plt.imshow(testimg)
# plt.show()

# data = img_to_array(testimg)
# data = np.expand_dims(data, axis=0)
# data = preprocess_input(data)
# predicts = model.predict(data)
# results = decode_predictions(predicts, top=5)[0]
# for r in results:
#     name = r[1]
#     pct = r[2]
#     print(f"これは、「{name}」です。（{pct:.1%})")

filenames=[r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img1.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img2.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img3.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img4.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img5.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img6.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img7.jpg",
           r"C:\Users\1612h\Deep_Learning\python3nenn\CNN\7_test_images\img8.jpg"]
img=[]
plt.figure(figsize=(16,10))
for i, filename in enumerate(filenames):
    img.append(load_img(filename, target_size=(224,224)))
    data=img_to_array(img[i])
    data=np.expand_dims(data, axis=0)
    data=preprocess_input(data)
    predicts=model.predict(data)
    results=decode_predictions(predicts, top=5)[0]
    plt.subplot(2,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i])

    for i,r in enumerate(results):
        name=r[1]
        pct=r[2]
        msg=f"{name}({pct:.1%})"
        plt.text(20, 250+i*16,msg)
plt.show()