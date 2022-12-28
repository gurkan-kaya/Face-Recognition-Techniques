#FaceNet and Deepface by using Deepface Library

from deepface import DeepFace
import cv2

###model_name = "Facenet512", "OpenFace", "DeepFace", "VGG-Face", "Facenet"
###detector_backend = "ssd", "opencv","mtcnn", "retinaface", "dlib"
###distance_metric =  "euclidean", "euclidean_l2","cosine"

#####################################################################################
#yüz tespiti ve hizalama
detected_face = DeepFace.detectFace(img_path="dataset/test/bayraktartest1.webp",
                                     detector_backend="opencv")
detected_face = cv2.cvtColor(detected_face,cv2.COLOR_BGR2RGB)
cv2.imshow("Yüz Tespit Edildi",detected_face)
#####################################################################################
#yüz doğrulama
verified_face = DeepFace.verify(img1_path="dataset/test/bayraktartest1.webp",
                                     img2_path="dataset/test/bayraktartest2.webp",
                                     detector_backend="opencv",
                                     model_name="DeepFace",
                                     distance_metric="euclidean")
 
print(verified_face)
#####################################################################################
#yüz tanıma
recognized_face = DeepFace.find(img_path="dataset/test/sancartest1.webp",
                                      db_path="dataset/train",
                                      detector_backend="opencv",
                                      model_name="Facenet",
                                      distance_metric="euclidean")
  
print(recognized_face)
#####################################################################################
#yüz analizi
analyzed_face = DeepFace.analyze(img_path="dataset/test/sancartest1.webp",
                                      actions=['emotion','age','gender','race'])
  
print(analyzed_face)
#####################################################################################

