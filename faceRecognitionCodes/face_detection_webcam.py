# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:29:40 2022

@author: kaya-
"""

import face_recognition
import cv2

# webcam görüntüsü yakalanır
webcam_video_capture = cv2.VideoCapture(0)

# Yüzlerin konumları için boş bir array tanımladık
faces_locations = []

#videodaki her frame için döngü oluşturduk
while True:
    #videodaki current frame image olarak alınır
    ret,current_frame = webcam_video_capture.read()
    
    #işlemi hızlandırmak için 1/4 oranında küçültme yaptık
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    
    #paramtrelerimiz image, no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    
    #farklı yüzler için döngü
    for index,current_face_location in enumerate(all_face_locations):
        #yüzün konum değerlerini bulmak için tuple'ın bölünmesi        
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        #doğru konum değerleri için 4 ile çarpıyoruz
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        #konum değerlerinin yazdırılması
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        #bulunan yüzlerin dikdörtgen içine alınması
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    #yüzün dikdörtgen içinde gösterilmesi
    cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_capture.release()
cv2.destroyAllWindows()   
