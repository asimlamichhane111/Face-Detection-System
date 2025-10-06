from deepface import DeepFace
import cv2

img1="D:/45 days/42/Face-Detection-OpenCV/faceDetection/e1.webp"
img2="D:/45 days/42/Face-Detection-OpenCV/faceDetection/e2.webp"

result=DeepFace.verify(img1,img2)
print("Is verified: ",result['verified'])
print("\nFull result: \n",result)