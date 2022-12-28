# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:55:20 2022

@author: kaya-
"""

#gerekli kütüphanelerin eklenmesi
import cv2
import face_recognition

#kamera görüntüsünün yakalanması
camera_stream = cv2.VideoCapture(0)

#ornek gorsellerle ilgili islemler
sancar_image = face_recognition.load_image_file('train/aziz_sancar.png')
sancar_face_encodings = face_recognition.face_encodings(sancar_image)[0]

bayraktar_image = face_recognition.load_image_file('train/selcuk_bayraktar.jpg')
bayraktar_face_encodings = face_recognition.face_encodings(bayraktar_image)[0]

gurkan_image = face_recognition.load_image_file('train/gurkan.jfif')
gurkan_face_encodings = face_recognition.face_encodings(gurkan_image)[0]

#yüz encodingleri ile aynı sırada yüzlere etiket verilmesi
face_encodings_known = [sancar_face_encodings, bayraktar_face_encodings, gurkan_face_encodings]
face_names = ["Aziz Sancar", "Selcuk Bayraktar", "Gurkan Kaya"]


#arrayler olusturuluyor
face_locations = []
face_encodings = []

#her framein incelenmesi için döngü
while True:
    #framein image olarak alınması
    ret,cur_frame = camera_stream.read()
    #frame 1/4 oranında küçültülüyor, daha hızlı işlem için
    cur_frame_small = cv2.resize(cur_frame,(0,0),fx=0.25,fy=0.25)
    
    #yüzler tespit ediliyor, parametreler: image,no_of_times_to_upsample, model, model cnn olursa daha doğru ama yavaş olur
    face_locations = face_recognition.face_locations(cur_frame_small,number_of_times_to_upsample=1,model='hog')
    
    #tüm yüzler için encoding
    face_encodings = face_recognition.face_encodings(cur_frame_small,face_locations)


    #face location ve embeddingleri döngü ile dönüyoruz. face embedding yüz verilerinin sayısal hali olarak düşünülebilir
    for current_face_location,current_face_encoding in zip(face_locations,face_encodings):
        #yüz konumunun 4 ayrı taraftan verisi
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #4 ile çarparak baştaki gerçek sizeı elde ediyoruz
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #eşleşmeler
        matches = face_recognition.compare_faces(face_encodings_known, current_face_encoding)
       
        person_name = 'Bilinmeyen Yüz'
        
        #bulunan eşleşmelere göre isim yüz eşleştirmesi
        if True in matches:
            first_match_index = matches.index(True)
            person_name = face_names[first_match_index]
        
        #yüz etrafına dikdörtgen çizilmesi 
        cv2.rectangle(cur_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        #isim yazılması
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(cur_frame, person_name, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    

    cv2.imshow("Kamera",cur_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera_stream.release()
cv2.destroyAllWindows()        










