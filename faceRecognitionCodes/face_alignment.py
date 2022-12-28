
#ilgili kütüphaneler
import cv2
import dlib

#kamera görüntüsü
kamera = cv2.VideoCapture(0)

#dlib'in HOG + Linear SVM yüz tespit modeli
face_detection = dlib.get_frontal_face_detector()

# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
face_shape_predictor = dlib.shape_predictor('model/face_landmarks.dat')

#yüz konumları arrayi
yuz_konum = []

#her frame için döngü
while True:
    face_landmarks = dlib.full_object_detections()
    ret,current_frame = kamera.read()

    yuz_konum = face_detection(current_frame,1)    

    for index,cur_face_location in enumerate(yuz_konum):     
        face_landmarks.append(face_shape_predictor(current_frame, cur_face_location))
    
    # yüzü dik konuma getirir
    face_chips = dlib.get_face_chips(current_frame,face_landmarks) 

    for index,current_face_chip in enumerate(face_chips):
        cv2.imshow("Face no "+str(index+1),current_face_chip)
    
    cv2.imshow("Face Alignment",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()        










