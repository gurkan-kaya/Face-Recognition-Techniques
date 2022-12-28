# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:51:51 2022

@author: kaya-
"""

#ilgili kütüphaneler
import cv2
import face_recognition

#test için kullanılacak görsel
image = cv2.imread('test/sancar_bayraktar.webp')

#örnek görsellerin yüklenmesi
sancar_image = face_recognition.load_image_file('train/aziz_sancar.png')
sancar_face_encodings = face_recognition.face_encodings(sancar_image)[0]
bayraktar_image = face_recognition.load_image_file('train/selcuk_bayraktar.jpg')
bayraktar_face_encodings = face_recognition.face_encodings(bayraktar_image)[0]
#yüz encodingleri ile aynı sırada yüzlere etiket verilmesi
known_face_encodings = [sancar_face_encodings, bayraktar_face_encodings]
face_names = ["Aziz Sancar" , "Selcuk Bayraktar"]
#iki yüzün de bulunduğu örnek görsel
image_to_recognize = face_recognition.load_image_file('test/sancar_bayraktar.webp')
#parametreler image,no_of_times_to_upsample, model . TÜm yüzler hog ile tespit ediliyor
all_face_locations = face_recognition.face_locations(image_to_recognize,model='hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location    
    #eşlelmelerin bulunması
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)   
    #tanınmayan yüzler için
    person_name = 'Tanınmayan Yüz'    
    #bulunan eşleşmelere göre isim yüz eşleştirmesi
    if True in all_matches:
        first_match_num = all_matches.index(True)
        person_name = face_names[first_match_num]    
    cv2.rectangle(image,(left_pos,top_pos),(right_pos,bottom_pos),(255,255,255),2)    
    #yüzlerin altında isimlerin yazdırılması
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, person_name, (left_pos,bottom_pos), font, 0.5, (255,0,0),2)
    
    cv2.imshow("Bulunan Yuzler",image)



