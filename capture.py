import cv2
import numpy as np
import matplotlib.pyplot as plot

cap = cv2.VideoCapture('d:/putin.mp4')

data_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(data_path)
num = 0
cap_num = 200
path_name = 'D:/train_pics/putin'
while(1):
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)
            img_name = '%s/%d.jpg ' % (path_name, num)
            cropped_image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            cv2.imwrite(img_name, cropped_image)
            num += 1
        if num > cap_num:
            break

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()