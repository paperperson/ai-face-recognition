import cv2

image_path = 'D:/train_pics/trump/ia_10460.jpg'



image = cv2.imread(image_path)

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image,1.3,5)
for(x,y,w,h) in faces:
   cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

cv2.imshow('title', image)
cv2.waitKey(0)
cv2.destroyAllWindows()