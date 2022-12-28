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
recognize_image = face_recognition.load_image_file('test/sancar_bayraktar.webp')
face_compare = face_recognition.face_encodings(recognize_image)[0]

#face distance bulma işlemi
face_distances = face_recognition.face_distance(known_face_encodings, face_compare)

for i,face_distance in enumerate(face_distances):
    print("{} isimli kişi için hesaplanan face distance değeri {:.1}".format(face_names[i],face_distance))