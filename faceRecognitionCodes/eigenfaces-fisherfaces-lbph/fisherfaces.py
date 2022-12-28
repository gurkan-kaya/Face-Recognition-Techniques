# pip install --user opencv-contrib-python

#ilgili kütüphanelerin import edilmesi
import cv2
import numpy as np
import os
from sys import exit


# face detection için yazılan metot
def face_detection(image_to_detect):
    
    #eigenface ve fisherface için gerektiğinden dolayı grayscale dönüşümü yapılır
    gray_image = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)
    
    #Face Detection için  daha önceden eğitilmiş olan modeli kullanıyoruz.
    #LBPH'de iyi sonuç almak için lbpcascade_frontalface.xml kullanılmalıdır.
    #Eigenface ve Fisherface'te iyi sonuç almak için haarcascade_frontalface_default.xml kullanılmalıdır.
    f_detection_classifier = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    
    # classifier kullanılarak yüzlerin konumları tespit edilir
    face_locations_all = f_detection_classifier.detectMultiScale(gray_image)
    
    # hiçbir yüz bulunamazsa None dönülür
    if (len(face_locations_all) == 0):
        return None, None
    
    #tuple konum için 4e bölünür
    x,y,width,height = face_locations_all[0]
    
    #koordinatlar hesaplanır
    coordinates_of_face = gray_image[y:y+width, x:x+height]
    
    #eigenfaces ve fisherfaces algoritmaları için görseller aynı boyutta olmalıdır bu yüzden resize edilir. Bu adım LBPH için zorunlu değildir.
    coordinates_of_face = cv2.resize(coordinates_of_face,(500,500))
    
    #tespit edilen yüz ve konumu döndürülür
    return coordinates_of_face, face_locations_all[0]


# eğitim verisinin hazırlanması
def prepare_data(image_dir, label_index):
    
    #yüzlerin ve etiketlerin tutulması için array
    coordinates_of_faces = []
    labels_index = []
    
    #directory'deki görsellerin dosya isimleri
    images = os.listdir(image_dir)
    
    for image in images:
        image_path = image_dir + "/" + image
        train_image = cv2.imread(image_path)
        #eğitim için kullanılan her görseli propgram çalıştığında sırayla gösteriyoruz, bu adım optionaldır.
        cv2.imshow(names[label_index], cv2.resize(train_image,(500,500)))
        #görseller arası geçişte 0.1 saniye duraklanıyor
        cv2.waitKey(100)
        
        #yüz tespiti için yazdığımız metot burada kullanılıyor
        coordinates_of_face, box_coordinates = face_detection(train_image)
        
        if coordinates_of_face is not None:
            # fonksiyondan dönen yüzler listeye ekleniyor
            coordinates_of_faces.append(coordinates_of_face)
            labels_index.append(label_index)
    
    return coordinates_of_faces, labels_index

# Eğitim verilerinin hazırlanması adımı
names = []

names.append("Selcuk Bayraktar")
coordinates_of_face_bayraktar, labels_index_bayraktar = prepare_data("dataset/train/bayraktar",0)
    
names.append("Aziz Sancar")
coordinates_of_face_sancar, labels_index_sancar = prepare_data("dataset/train/sancar",1)
    
coordinates_of_face = coordinates_of_face_bayraktar + coordinates_of_face_sancar
labels_index = labels_index_bayraktar + labels_index_sancar

#toplam yüz sayısı ve isim sayısı yazdırılıyor
print("Yuz Sayisi:", len(coordinates_of_face))
print("Isim Sayisi:", len(names))


#Eğitim Adımı

#kullanılacak algoritmaya göre eğitim işlemi yapılıyor
f_classifier = cv2.face.FisherFaceRecognizer_create()


f_classifier.train(coordinates_of_face,np.array(labels_index))


#Test - Tahmin Adımı
classify_image = cv2.imread("dataset/test/sancartest2.jfif")

#resmin kopyası alınıyor
classify_image_copy = classify_image.copy()

#resimdeki yüz bulunuyor
coordinates_of_face_classify, rect_locations = face_detection(classify_image_copy)  

#yüz bulunamazsa
if coordinates_of_face_classify is None:
    print("Sınıflandırılacak yüz bulunamadı")
    exit()
    
#tahminleme işlemi
name_index, distance = f_classifier.predict(coordinates_of_face_classify)
name = names[name_index]
distance = abs(distance)

#yüzün çerçevelenmesi ve tahmin edilen ismin yazılması
(x,y,w,h) = rect_locations
cv2.rectangle(classify_image,(x,y),(x+w, y+h),(0,255,0),2)
cv2.putText(classify_image,name,(x,y-5),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),2)

#pencerede sonucun gösterimi yapılıyor
cv2.imshow("Fisherfaces Tahmin "+name, cv2.resize(classify_image, (500,500)))
cv2.waitKey(0)
cv2.destroyAllWindows()



