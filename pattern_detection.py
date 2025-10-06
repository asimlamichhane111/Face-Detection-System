import cv2
import mediapipe as mp
mp_face=mp.solutions.face_mesh
face_mesh=mp_face.FaceMesh()
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=face_mesh.process(rgb)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                h,w,c=frame.shape
                x,y=int(lm.x*w),int(lm.y*h)
                cv2.circle(frame,(x,y),1,(0,255,0),-1)
    cv2.imshow("Landmarks",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()