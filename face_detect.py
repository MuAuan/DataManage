# -*- coding:utf-8 -*-

import cv2
import numpy as np
import os.path

# 先ほど集めてきた画像データのあるディレクトリ
#input_data_path = './train_images0'

# 切り抜いた画像の保存先ディレクトリ(予めディレクトリを作っておいてください)
#save_path = './train_images_face0'

#OpenCVのデフォルトの分類器のpath。
#cascade_path = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"  #のファイルを使う)
#cascade_path = '/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
#cascade_path = "./lib/haarcascade_frontalface_default.xml"
cascade_path ="/Users/tosio/fastText/path/to/corpus/AA/.lib/haarcascade_frontalface_default.xml"
#cascade_path = "./lib/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

# 収集した画像の枚数(任意で変更)
image_count = 140
j=17 #train_images2

# 顔検知に成功した数(デフォルトで0を指定)
face_detect_count = 0

# 集めた画像データから顔が検知されたら、切り取り、保存する。
for i in range(image_count):
    print(i,image_count)
    if os.path.isfile("./train_images"+str(j)+"/" + str(i) + '.jpg'):  #(input_data_path + str(i) + '.jpg')+
        img = cv2.imread("./train_images"+str(j)+"/" + str(i) + '.jpg', cv2.IMREAD_COLOR)
        print(img)
        #img = cv2.imread(str(i) + '.jpg', cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(i,gray)
        face = faceCascade.detectMultiScale(gray, 1.1, 3)
        #face = faceCascade.detectMultiScale(img, 1.1, 3)
        #face = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1,minSize=(1, 1))

        if len(face) > 0:
            for rect in face:
        # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
        # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
        # cv2.imwrite('detected.jpg', img)
                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]

            #cv2.imwrite(save_path + 'face' + str(face_detect_count) + '.jpg', img[y:y+h, x:x+w])
            cv2.imwrite("./train_images_face" +str(j) + "/face" + str(face_detect_count) + '.jpg', img[y:y+h, x:x+w])
            face_detect_count = face_detect_count + 1
        else:
            print('image' + str(i) + ':No Face')
    else:
          print('image' + str(i) + ':No File')

