# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:38:39 2022

@author: kaya-
"""

# Gerekli kütüphaneleri import ediyoruz
import cv2
import face_recognition

# Yüz tespiti yapılacak görseli yoluyla birlikte tanımlıyoruz
face_detection_image = cv2.imread('images/sancar_bayraktar.jpg')

# Görsel doğru bir şekilde gözüküyor mu diye test ediyoruz.
#cv2.imshow("test", face_detection_image)

# Yüzlerin konumlarını bulan face_locations fonksiyonuna model olarak hog veriyoruz
# faces_locations= face_recognition.face_locations(face_detection_image,model="hog")

# Kaç tane yüz bulunduğunu yazdırıyoruz
#print("Fotoğrafta HoG modeli ile {} adet yüz tespit edilmiştir".format(len(faces_locations)))

# Yüzlerin konumlarını bulan face_locations fonksiyonuna model olarak cnn veriyoruz
faces_locations= face_recognition.face_locations(face_detection_image,model="cnn")

# Kaç tane yüz bulunduğunu yazdırıyoruz
#print("Fotoğrafta CNN modeli ile {} adet yüz tespit edilmiştir".format(len(faces_locations)))

# HoG ile tespit edilen yüzlerin konumları
# for index,current_face_location in enumerate(faces_locations):
#     # Dört ayrı taraftan da konumları alıyoruz
#     top, right, bottom, left = current_face_location
#     print("{} numaralı yüzün konumları. top:{}, right:{}, bottom{}, left{}".format(index+1,  top, right, bottom, left))
#     # Tespit edilen yüzler kırpılır
#     face_image = face_detection_image[top:bottom,left:right]
#     # Yüzler ayrı pencerelerde gösterilir
#     cv2.imshow(str(index) + " numaralı yüz ", face_image)

# # CNN ile tespit edilen yüzlerin konumları
for index,current_face_location in enumerate(faces_locations):
    # Dört ayrı taraftan da konumları alıyoruz
    top, right, bottom, left = current_face_location
    print("{} numaralı yüzün konumları. top: {}, right: {}, bottom: {}, left: {}".format(index+1,  top, right, bottom, left))
    face_image = face_detection_image[top:bottom,left:right]
    # Yüzler ayrı pencerelerde gösterilir
    cv2.imshow(str(index) + " numaralı yüz ", face_image)

























