import cv2
from deepface import DeepFace
database={
    "Asim":"Asim.jpg",
    "Subash":"Subash.jpg",
}
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imwrite("current_frame.jpg",frame)
    for name,img_path in database.items():
        result=DeepFace.verify('current_frame.jpg',img_path,enforce_detection=False)
        if result['verified']:
            cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            break
    cv2.imshow("Face Recognition",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()