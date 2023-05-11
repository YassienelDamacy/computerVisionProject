import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mphands=mp.solutions.hands
hands=mphands.Hands()
mpFaceDetection = mp.solutions.face_detection
faceDetecition = mpFaceDetection.FaceDetection()
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) 
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)

while(True):
    ret, img = cap.read()
#hand tracking
    img=cv2.cvtColor(cv2.flip(img,1),cv2.COLOR_BGR2RGB)
    results=hands.process(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,
            hand_landmarks,
            mphands.HAND_CONNECTIONS)
#face tracking
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetecition.process(imgRGB)
    if results.detections:
        for  detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw , ic = img.shape
            bbox = int(bboxC.xmin * iw ) , int(bboxC.ymin * ih ), \
                   int(bboxC.width * iw ), int(bboxC.height * ih )   
            cv2.rectangle(img,bbox,(255,0,255),2)
#pose tracking
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks :
        mpDraw.draw_landmarks(img,results.pose_landmarks,
        mpPose.POSE_CONNECTIONS)
        for lm in results.pose_landmarks.landmark:
            h,w,c = img.shape
            cx , cy = int(lm.x * w) , int(lm.y * h)
            cv2.circle(img,(cx,cy),1,(255,0,0), cv2.FILLED)
#face mesh
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img , faceLms ,
            mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)

    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()